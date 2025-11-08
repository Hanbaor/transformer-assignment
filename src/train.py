import os
import yaml
import time
import math
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from model import Encoder, Decoder, Transformer
from dataset import get_data_loaders_and_vocabs
from torch.amp import autocast, GradScaler

try:
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_epoch(model, iterator, optimizer, criterion, clip, scaler):
    model.train()
    epoch_loss = 0
    pbar = tqdm(iterator, desc="Training")
    for batch in pbar:
        src, tgt = batch
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        
        optimizer.zero_grad()

        with autocast('cuda', enabled=(model.device.type == 'cuda')):
            output, _ = model(src, tgt[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_flat = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt_flat)

        if scaler is not None and model.device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        pbar = tqdm(iterator, desc="Evaluating")
        for batch in pbar:
            src, tgt = batch
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            with autocast('cuda', enabled=(model.device.type == 'cuda')):
                output, _ = model(src, tgt[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_flat = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt_flat)

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    return epoch_loss / len(iterator)


def _is_triton_available():
    try:
        import triton
        import triton.language as tl
        return True
    except Exception:
        return False


def main(cfg):
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, _, vocab_transform, PAD_IDX = get_data_loaders_and_vocabs(cfg)
    
    SRC_VOCAB_SIZE = len(vocab_transform[cfg['src_language']])
    TGT_VOCAB_SIZE = len(vocab_transform[cfg['tgt_language']])

    enc = Encoder(SRC_VOCAB_SIZE, cfg['d_model'], cfg['n_encoder_layers'], cfg['n_heads'], cfg['d_ff'], cfg['dropout'])
    dec = Decoder(TGT_VOCAB_SIZE, cfg['d_model'], cfg['n_decoder_layers'], cfg['n_heads'], cfg['d_ff'], cfg['dropout'])

    model = Transformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
    model.apply(initialize_weights)

    # 可选：PyTorch 2.0+ 编译优化（Windows/Triton不可用时自动回退）
    if cfg.get('compile', False):
        backend = cfg.get('compile_backend', None)
        if backend is None:
            backend = 'aot_eager' if (os.name == 'nt' or not _is_triton_available()) else 'inductor'
        try:
            model = torch.compile(model, mode=cfg.get('compile_mode', 'reduce-overhead'), backend=backend)
            print(f"[speed] torch.compile enabled (backend={backend})")
        except Exception as e:
            print(f"[speed] torch.compile disabled: {e}")

    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 优化器：AdamW（若 CUDA 支持 fused）
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['lr'],
            fused=True if device.type == 'cuda' else False
        )
        print(f"[speed] using AdamW (fused={device.type == 'cuda'})")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        print("[speed] using AdamW (no fused)")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    best_valid_loss = float('inf')

    # 记录每轮损失与 PPL 以便绘图
    train_losses = []
    valid_losses = []
    train_ppls = []
    valid_ppls = []

    for epoch in range(cfg['epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg['grad_clip'], scaler)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()

        # 记录到曲线
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_ppls.append(math.exp(train_loss))
        valid_ppls.append(math.exp(valid_loss))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if not os.path.exists(cfg['output_dir']):
                os.makedirs(cfg['output_dir'])
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], cfg['model_save_name']))
        
        print(f'Epoch: {epoch+1:02} | Time: {end_time - start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # 训练结束绘制曲线
    try:
        import matplotlib.pyplot as plt
        out_dir = cfg['output_dir']
        os.makedirs(out_dir, exist_ok=True)

        # Loss 曲线
        loss_path = os.path.join(out_dir, cfg.get('plot_save_name', 'training_curves.png'))
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        print(f"[plot] Saved training curves to {loss_path}")

        #PPL 曲线
        ppl_path = os.path.join(out_dir, cfg.get('plot_ppl_save_name', 'training_ppl.png'))
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(epochs, train_ppls, label='Train PPL')
        plt.plot(epochs, valid_ppls, label='Valid PPL')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training & Validation Perplexity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ppl_path, dpi=150)
        print(f"[plot] Saved PPL curves to {ppl_path}")
    except Exception as e:
        print(f"[plot] Skipped plotting: {e}. Install matplotlib with: pip install matplotlib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Transformer model from scratch.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)