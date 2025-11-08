import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET
from collections import Counter

try:
    import spacy
except Exception:
    spacy = None

# 特殊符号与索引
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


# 简易 Vocab 类
class Vocab:
    def __init__(self, itos):
        self._itos = list(itos)
        self._stoi = {tok: idx for idx, tok in enumerate(self._itos)}
        self._default_index = UNK_IDX

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, token):
        return self._stoi.get(token, self._default_index)

    def get_itos(self):
        return self._itos

    def set_default_index(self, idx):
        self._default_index = idx

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def __call__(self, tokens):
        return self.lookup_indices(tokens)


def build_vocab_from_iterator(iterator, min_freq, specials, special_first=True, log_every=0, name=None):
    counter = Counter()
    sent_count = 0
    for tokens in iterator:
        counter.update(tokens)
        sent_count += 1
        if log_every and sent_count % log_every == 0:
            print(f"[vocab] {name or ''} processed {sent_count} sentences...")
    # 过滤低频与特殊符号
    tokens = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq and tok not in specials]
    tokens.sort(key=lambda x: (-x[1], x[0]))
    itos = (list(specials) + [tok for tok, _ in tokens]) if special_first else ([tok for tok, _ in tokens] + list(specials))
    vocab = Vocab(itos)
    vocab.set_default_index(UNK_IDX)
    print(f"[vocab] {name or ''} done. size={len(vocab)} (processed {sent_count} sentences)")
    return vocab


def save_vocabs(vocab_transform, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    for ln, vocab in vocab_transform.items():
        path = os.path.join(save_dir, f'vocab_{ln}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"itos": vocab.get_itos()}, f, ensure_ascii=False)
    meta_path = os.path.join(save_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "src_language": cfg['src_language'],
            "tgt_language": cfg['tgt_language'],
            "tokenizer": cfg.get('tokenizer', 'spacy_blank'),
            "vocab_min_freq": int(cfg.get('vocab_min_freq', 1)),
        }, f, ensure_ascii=False)


def load_vocabs(save_dir, cfg):
    try:
        meta_path = os.path.join(save_dir, 'meta.json')
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # 严格匹配关键配置，避免错配
        if (meta.get('src_language') != cfg.get('src_language') or
            meta.get('tgt_language') != cfg.get('tgt_language') or
            str(meta.get('tokenizer', '')).lower() != str(cfg.get('tokenizer', '')).lower() or
            int(meta.get('vocab_min_freq', 1)) != int(cfg.get('vocab_min_freq', 1))):
            return None

        vocabs = {}
        for ln in [cfg['src_language'], cfg['tgt_language']]:
            path = os.path.join(save_dir, f'vocab_{ln}.json')
            if not os.path.isfile(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            itos = data.get('itos')
            if not itos:
                return None
            v = Vocab(itos)
            v.set_default_index(UNK_IDX)
            vocabs[ln] = v
        return vocabs if len(vocabs) == 2 else None
    except Exception:
        return None


# 数据集：接收已配对的 (src, tgt) 文本
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_tokenizer(lang, mode='spacy_blank'):
    mode = (mode or 'spacy_blank').lower()
    if mode == 'space' or spacy is None:
        return lambda s: s.lower().split()

    if mode == 'spacy_blank':
        nlp = spacy.blank('de' if lang == 'de' else 'en')
        return lambda s: [t.text.lower() for t in nlp(s)]

    if mode == 'spacy_model':
        if lang == 'de':
            nlp = spacy.load('de_core_news_sm', disable=['tagger', 'parser', 'ner'])
        elif lang == 'en':
            nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        else:
            raise ValueError(f'Unsupported language: {lang}')
        return lambda s: [t.text.lower() for t in nlp(s)]

    raise ValueError(f'Unsupported tokenizer mode: {mode}')


def read_train_pairs(base_dir, src_lang, tgt_lang):

    src_path = os.path.join(base_dir, f'train.tags.en-de.{src_lang}')
    tgt_path = os.path.join(base_dir, f'train.tags.en-de.{tgt_lang}')
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Missing train files: {src_path} or {tgt_path}")
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_lines = [ln.strip() for ln in f_src if ln.strip() and not ln.strip().startswith('<')]
        tgt_lines = [ln.strip() for ln in f_tgt if ln.strip() and not ln.strip().startswith('<')]
    assert len(src_lines) == len(tgt_lines), f"Train src/tgt size mismatch: {len(src_lines)} vs {len(tgt_lines)}"
    return list(zip(src_lines, tgt_lines))


def read_xml_pairs(base_dir, split_prefix, src_lang, tgt_lang):
    src_xml = os.path.join(base_dir, f'IWSLT17.TED.{split_prefix}.en-de.{src_lang}.xml')
    tgt_xml = os.path.join(base_dir, f'IWSLT17.TED.{split_prefix}.en-de.{tgt_lang}.xml')
    if not os.path.exists(src_xml) or not os.path.exists(tgt_xml):
        raise FileNotFoundError(f"Missing XML files: {src_xml} or {tgt_xml}")

    def extract_segs(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return [seg.text.strip() for seg in root.iter('seg') if seg.text and seg.text.strip()]

    src_segs = extract_segs(src_xml)
    tgt_segs = extract_segs(tgt_xml)
    assert len(src_segs) == len(tgt_segs), f"{split_prefix} src/tgt size mismatch: {len(src_segs)} vs {len(tgt_segs)}"
    return list(zip(src_segs, tgt_segs))


def get_data_loaders_and_vocabs(cfg):

    print("[data] initializing...")

    SRC_LANGUAGE = cfg['src_language']  
    TGT_LANGUAGE = cfg['tgt_language'] 
    tokenizer_mode = cfg.get('tokenizer', 'spacy_blank')
    max_seq_len = int(cfg.get('max_seq_len', 100))  
    body_max_len = max(0, max_seq_len - 2)          
    num_workers = int(cfg.get('num_workers', 0))    
    pin_memory = (cfg.get('device', 'cuda').lower() == 'cuda')

    # 相对当前文件，定位到/data/en-de
    project_root = os.path.dirname(os.path.dirname(__file__))  
    data_dir = os.path.join(project_root, 'data', 'en-de')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    print(f"[data] tokenizer mode: {tokenizer_mode}")
    token_transform = {
        SRC_LANGUAGE: make_tokenizer(SRC_LANGUAGE, tokenizer_mode),
        TGT_LANGUAGE: make_tokenizer(TGT_LANGUAGE, tokenizer_mode),
    }

    # 词表缓存目录
    min_freq = max(1, int(cfg.get('vocab_min_freq', 1)))
    vocab_root = cfg.get('vocab_cache_dir', os.path.join(cfg.get('output_dir', 'results'), 'vocabs'))
    vocab_dir = os.path.join(vocab_root, f"{SRC_LANGUAGE}-{TGT_LANGUAGE}", f"{tokenizer_mode}_minfreq{min_freq}")

    # 尝试加载缓存词表
    vocab_transform = load_vocabs(vocab_dir, cfg)
    if vocab_transform:
        print(f"[vocab] loaded cached vocabs from {vocab_dir}")
    else:
        print(f"[vocab] no cache at {vocab_dir}; will build and cache.")

    print("[data] reading train pairs...")
    train_pairs = read_train_pairs(data_dir, SRC_LANGUAGE, TGT_LANGUAGE)
    print(f"[data] train size: {len(train_pairs)}")

    debug_n = cfg.get('debug_small_train', None)
    if isinstance(debug_n, int) and debug_n > 0:
        train_pairs = train_pairs[:debug_n]
        print(f"[data] debug mode: using first {len(train_pairs)} train pairs")

    print("[data] reading dev pairs (dev2010)...")
    val_pairs = read_xml_pairs(data_dir, 'dev2010', SRC_LANGUAGE, TGT_LANGUAGE)
    print(f"[data] dev size: {len(val_pairs)}")

    print("[data] reading test pairs (tst2014)...")
    test_pairs = read_xml_pairs(data_dir, 'tst2014', SRC_LANGUAGE, TGT_LANGUAGE)
    print(f"[data] test size: {len(test_pairs)}")

    # 若缓存不存在，构建词表并保存
    if not vocab_transform:
        def yield_tokens(pairs, language):
            idx = 0 if language == SRC_LANGUAGE else 1
            for src_sample, tgt_sample in pairs:
                text = src_sample if idx == 0 else tgt_sample
                yield token_transform[language](text)

        vocab_transform = {}
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            print(f"[vocab] building {ln} vocab (this may take a while)...")
            vocab_transform[ln] = build_vocab_from_iterator(
                yield_tokens(train_pairs, ln),
                min_freq=min_freq,
                specials=special_symbols,
                special_first=True,
                log_every=100000,
                name=ln
            )
            vocab_transform[ln].set_default_index(UNK_IDX)
            print(f"[vocab] {ln} size: {len(vocab_transform[ln])}")

        save_vocabs(vocab_transform, vocab_dir, cfg)
        print(f"[vocab] saved vocabs to {vocab_dir}")

    # 批处理：tokenize -> vocab 索引 -> 加 BOS/EOS -> 截断 -> pad
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_tokens = token_transform[SRC_LANGUAGE](src_sample)[:body_max_len]
            tgt_tokens = token_transform[TGT_LANGUAGE](tgt_sample)[:body_max_len]

            src_idx = [BOS_IDX] + vocab_transform[SRC_LANGUAGE].lookup_indices(src_tokens) + [EOS_IDX]
            tgt_idx = [BOS_IDX] + vocab_transform[TGT_LANGUAGE].lookup_indices(tgt_tokens) + [EOS_IDX]

            if len(src_idx) > max_seq_len:
                src_idx = src_idx[:max_seq_len]
            if len(tgt_idx) > max_seq_len:
                tgt_idx = tgt_idx[:max_seq_len]

            src_batch.append(torch.tensor(src_idx, dtype=torch.long))
            tgt_batch.append(torch.tensor(tgt_idx, dtype=torch.long))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, tgt_batch

    print("[data] creating dataloaders...")
    train_dataloader = DataLoader(
        TranslationDataset(train_pairs),
        batch_size=cfg['batch_size'],
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        TranslationDataset(val_pairs),
        batch_size=cfg['batch_size'],
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        TranslationDataset(test_pairs),
        batch_size=cfg['batch_size'],
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("[data] ready.")
    return train_dataloader, val_dataloader, test_dataloader, vocab_transform, PAD_IDX