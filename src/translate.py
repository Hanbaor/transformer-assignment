import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast

from model import Encoder, Decoder, Transformer
from dataset import get_data_loaders_and_vocabs, BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, read_xml_pairs


def _tokenize_sentence(sentence: str, cfg) -> list[str]:

    mode = str(cfg.get('tokenizer', 'space')).lower()
    src_lang = str(cfg.get('src_language', 'de')).lower()
    s = sentence.strip()

    if mode == 'space':
        return s.lower().split()

    try:
        import spacy
        if mode == 'spacy_blank':
            nlp = spacy.blank('de' if src_lang.startswith('de') else 'en')
        elif mode == 'spacy_model':
            model_name = 'de_core_news_sm' if src_lang.startswith('de') else 'en_core_web_sm'
            nlp = spacy.load(model_name)
        else:
            return s.lower().split()
        return [t.text.lower() for t in nlp(s)]
    except Exception:
        return s.lower().split()


def _indexes_to_tokens(indexes: list[int], vocab, drop_unk: bool = True) -> list[str]:

    itos = vocab.get_itos()
    tokens = []
    for i in indexes:
        if i in (BOS_IDX, EOS_IDX, PAD_IDX):
            continue
        if drop_unk and i == UNK_IDX:
            continue
        tokens.append(itos[i])
    return tokens


def _length_penalty(length: int, alpha: float) -> float:

    return ((5 + length) ** alpha) / (6 ** alpha) if alpha > 0 else 1.0


def beam_search_decode(memory, src_mask, model, device, max_len=50, beam_size=5, alpha=0.6):

    if beam_size <= 1:
        gen = [BOS_IDX]
        for _ in range(max_len):
            tgt = torch.LongTensor(gen).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = model.decoder(tgt, memory, tgt_mask, src_mask)
            next_tok = logits.argmax(2)[:, -1].item()
            gen.append(next_tok)
            if next_tok == EOS_IDX:
                break
        return gen

    beams = [([BOS_IDX], 0.0, False)]
    finished = []

    for _ in range(max_len):
        active = [(seq, score) for (seq, score, ended) in beams if not ended]
        if len(active) == 0:
            break

        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(seq) for seq, _ in active],
            batch_first=True
        ).to(device)
        tgt_mask = model.make_tgt_mask(tgt_batch)

        repeat_n = tgt_batch.size(0)
        mem = memory.repeat(repeat_n, 1, 1)
        smask = src_mask.repeat(repeat_n, 1, 1, 1)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model.decoder(tgt_batch, mem, tgt_mask, smask)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [repeat_n, vocab]

        candidates = []
        for i, (seq, score) in enumerate(active):
            topk_logp, topk_idx = torch.topk(log_probs[i], beam_size)
            for j in range(topk_idx.size(0)):
                tok = int(topk_idx[j].item())
                new_seq = seq + [tok]
                new_score = score + float(topk_logp[j].item())
                ended = (tok == EOS_IDX)
                norm = new_score / _length_penalty(len(new_seq), alpha)
                candidates.append((new_seq, new_score, ended, norm))

        candidates.sort(key=lambda x: x[3], reverse=True)
        beams = [(seq, score, ended) for (seq, score, ended, _) in candidates[:beam_size]]

        newly_finished = [(seq, score) for (seq, score, ended) in beams if ended]
        finished.extend(newly_finished)

        if len(finished) >= beam_size:
            break

    if len(finished) > 0:
        finished.sort(key=lambda x: x[1] / _length_penalty(len(x[0]), alpha), reverse=True)
        return finished[0][0]
    beams.sort(key=lambda x: x[1] / _length_penalty(len(x[0]), alpha), reverse=True)
    return beams[0][0]


def _unk_replace_with_attention(gen_indexes, src_tensor, memory, src_mask, model, tgt_vocab, src_vocab, device):

    tgt = torch.LongTensor(gen_indexes).unsqueeze(0).to(device)
    tgt_mask = model.make_tgt_mask(tgt)
    with torch.no_grad():
        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits, attention = model.decoder(tgt, memory, tgt_mask, src_mask)  # attention: [1, heads, tgt_len, src_len]

    itos_tgt = tgt_vocab.get_itos()
    itos_src = src_vocab.get_itos()
    attn_mean = attention.mean(dim=1)[0]  # [tgt_len, src_len]
    src_list = src_tensor.squeeze(0).tolist()

    tokens = []
    for pos, idx in enumerate(gen_indexes):
        if idx in (BOS_IDX, EOS_IDX, PAD_IDX):
            continue
        if idx == UNK_IDX:
            src_pos = int(attn_mean[pos].argmax().item())
            candidate_idx = src_list[src_pos]
            if candidate_idx in (BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX):
                topk = torch.topk(attn_mean[pos], k=min(5, attn_mean.size(-1))).indices.tolist()
                for sp in topk:
                    cand = src_list[int(sp)]
                    if cand not in (BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX):
                        candidate_idx = cand
                        break
            tokens.append(itos_src[candidate_idx])
        else:
            tokens.append(itos_tgt[idx])
    return tokens


def translate_sentence(sentence, cfg, src_vocab, tgt_vocab, model, device,
                       max_len=50, beam_size=5, length_penalty=0.6, drop_unk=True, unk_replace=False):

    model.eval()

    tokens = _tokenize_sentence(sentence, cfg)
    bos_tok = src_vocab.get_itos()[BOS_IDX]
    eos_tok = src_vocab.get_itos()[EOS_IDX]
    tokens = [bos_tok] + tokens + [eos_tok]

    src_indexes = [src_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        with autocast('cuda', enabled=(device.type == 'cuda')):
            memory = model.encoder(src_tensor, src_mask)

    gen_indexes = beam_search_decode(
        memory, src_mask, model, device,
        max_len=max_len, beam_size=beam_size, alpha=length_penalty
    )

    if unk_replace:
        tgt_tokens = _unk_replace_with_attention(gen_indexes, src_tensor, memory, src_mask, model, tgt_vocab, src_vocab, device)
        return tgt_tokens, None

    tgt_tokens = _indexes_to_tokens(gen_indexes, tgt_vocab, drop_unk=drop_unk)
    return tgt_tokens, None


def translate_loader(loader, src_vocab, tgt_vocab, model, device,
                     max_len=50, output_path=None, return_refs=False,
                     beam_size=5, length_penalty=0.6, drop_unk=True,
                     raw_pairs=None, unk_replace=False):

    model.eval()
    translations: list[str] = []
    references: list[str] = []
    sources: list[str] = []

    cursor = 0
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(loader, desc="Translating dataset"):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            bsz = src_batch.size(0)

            for b in range(bsz):
                src = src_batch[b:b+1]   # [1, src_len]
                tgt = tgt_batch[b]       # [tgt_len]

                # 原始源文本（避免左侧出现 <unk>）
                if raw_pairs is not None and cursor + b < len(raw_pairs):
                    src_text = raw_pairs[cursor + b][0]
                else:
                    src_tokens = _indexes_to_tokens(src.squeeze(0).tolist(), src_vocab, drop_unk=False)
                    src_text = " ".join(src_tokens)

                src_mask = model.make_src_mask(src)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    memory = model.encoder(src, src_mask)

                gen = beam_search_decode(
                    memory, src_mask, model, device,
                    max_len=max_len, beam_size=beam_size, alpha=length_penalty
                )

                if unk_replace:
                    hyp_tokens = _unk_replace_with_attention(gen, src, memory, src_mask, model, tgt_vocab, src_vocab, device)
                else:
                    hyp_tokens = _indexes_to_tokens(gen, tgt_vocab, drop_unk=drop_unk)

                hyp_text = " ".join(hyp_tokens)

                translations.append(hyp_text)
                sources.append(src_text)

                if return_refs:
                    ref_tokens = _indexes_to_tokens(tgt.tolist(), tgt_vocab, drop_unk=drop_unk)
                    references.append(" ".join(ref_tokens))

            cursor += bsz

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for src_text, hyp_text in zip(sources, translations):
                f.write(f"{src_text}\t{hyp_text}\n")
        print(f"[translate] Saved {len(translations)} src-hyp pairs to {output_path}")

    return translations, references if return_refs else None, sources


def compute_bleu(hypotheses: list[str], references: list[str]) -> float | None:

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='none')
        print(f"[eval] BLEU = {bleu.score:.2f}")
        return float(bleu.score)
    except Exception:
        print("[eval] sacrebleu not installed. Install with:")
        print("  pip install sacrebleu")
        return None


def main(cfg, sentence=None, dataset=None, output=None, eval=False,
         beam_size=5, length_penalty=0.6, keep_unk=False, unk_replace=False):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['device'] == 'cuda' else 'cpu')

    train_loader, val_loader, test_loader, vocab_transform, PAD_IDX_local = get_data_loaders_and_vocabs(cfg)

    SRC_VOCAB = vocab_transform[cfg['src_language']]
    TGT_VOCAB = vocab_transform[cfg['tgt_language']]

    enc = Encoder(
        len(SRC_VOCAB),
        cfg['d_model'],
        cfg['n_encoder_layers'],
        cfg['n_heads'],
        cfg['d_ff'],
        cfg['dropout']
    )
    dec = Decoder(
        len(TGT_VOCAB),
        cfg['d_model'],
        cfg['n_decoder_layers'],
        cfg['n_heads'],
        cfg['d_ff'],
        cfg['dropout']
    )
    model = Transformer(enc, dec, PAD_IDX_local, PAD_IDX_local, device).to(device)
    model_path = os.path.join(cfg['output_dir'], cfg['model_save_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))

    max_len = int(cfg.get('max_seq_len', 100))
    drop_unk = not keep_unk

    if dataset is not None:
        split = dataset.lower()
        if split in ("test", "tst2014"):
            loader = test_loader
            default_out = os.path.join(cfg['output_dir'], "test_pairs.txt")
            split_name = "test"
            # 读取原始测试集句子
            project_root = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(project_root, 'data', 'en-de')
            raw_pairs = read_xml_pairs(data_dir, 'tst2014', cfg['src_language'], cfg['tgt_language'])
        elif split in ("dev", "dev2010"):
            loader = val_loader
            default_out = os.path.join(cfg['output_dir'], "dev_pairs.txt")
            split_name = "dev"
            project_root = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(project_root, 'data', 'en-de')
            raw_pairs = read_xml_pairs(data_dir, 'dev2010', cfg['src_language'], cfg['tgt_language'])
        else:
            raise ValueError(f"Unknown dataset split: {dataset}. Use 'test'/'tst2014' or 'dev'/'dev2010'.")

        out_path = output if output is not None else default_out
        hyps, refs, _srcs = translate_loader(
            loader,
            SRC_VOCAB,
            TGT_VOCAB,
            model,
            device,
            max_len=max_len,
            output_path=out_path,
            return_refs=True,
            beam_size=beam_size,
            length_penalty=length_penalty,
            drop_unk=drop_unk,
            raw_pairs=raw_pairs,
            unk_replace=unk_replace
        )

        if eval and refs is not None and len(refs) == len(hyps):
            bleu = compute_bleu(hyps, refs)
            if bleu is not None:
                eval_path = os.path.join(cfg['output_dir'], f"{split_name}_bleu.txt")
                with open(eval_path, "w", encoding="utf-8") as f:
                    f.write(f"BLEU = {bleu:.2f}\n")
                print(f"[eval] Saved BLEU to {eval_path}")
        else:
            if eval:
                print("[eval] References not available; skip BLEU.")
        return

    # 单句翻译
    if sentence is None:
        raise ValueError("Provide --sentence for single translation or --dataset for batch translation.")

    translation, _ = translate_sentence(
        sentence, cfg, SRC_VOCAB, TGT_VOCAB, model, device,
        max_len=max_len, beam_size=beam_size, length_penalty=length_penalty,
        drop_unk=drop_unk, unk_replace=unk_replace
    )

    print("-" * 30)
    print(f"Source: {sentence}")
    print(f"Translated: {' '.join(translation)}")
    print("-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate: single sentence or dataset split with optional BLEU.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file.')
    parser.add_argument('--sentence', type=str, help='Sentence to translate.')
    parser.add_argument('--dataset', type=str, choices=['dev', 'test', 'dev2010', 'tst2014'], help='Dataset split to translate.')
    parser.add_argument('--output', type=str, help='Output file path for dataset translation (pairs).')
    parser.add_argument('--eval', action='store_true', help='Compute BLEU for dataset translation.')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search (>=1).')
    parser.add_argument('--length_penalty', type=float, default=0.6, help='Length penalty alpha (GNMT style).')
    parser.add_argument('--keep_unk', action='store_true', help='Keep <unk> tokens in outputs.')
    parser.add_argument('--unk_replace', action='store_true', help='Replace <unk> using cross-attention with source tokens.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(
        config,
        sentence=args.sentence,
        dataset=args.dataset,
        output=args.output,
        eval=args.eval,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        keep_unk=args.keep_unk,
        unk_replace=args.unk_replace
    )