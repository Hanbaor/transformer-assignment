import torch
import torch.nn as nn
import math

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 定义Q, K, V和最终输出的线性变换层
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性投影
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)

        # 拆分多头：[batch, len, heads, d_k] -> [batch, heads, len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        if mask is not None:
            # 统一为布尔掩码，填充为 -1e9
            if mask.dtype != torch.bool:
                mask = mask.bool()
            scores = scores.masked_fill(~mask, -1e9)

        #softmax
        attention = torch.softmax(scores.float(), dim=-1).to(V.dtype)
        attention = self.dropout(attention)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc_o(context)
        return out, attention


# Position-wise Feed-Forward Network)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


# 编码器层 (Encoder Layer)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))
        _src = self.ff(src)
        src = self.norm2(src + self.dropout(_src))
        return src


# 解码器层 (Decoder Layer)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))
        _tgt, attention = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt))
        _tgt = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt))
        return tgt, attention


# 正弦位置编码
def _build_sinusoidal_pe(max_len: int, d_model: int):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# 完整的编码器和解码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=512):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))

        self.register_buffer('pos_table', _build_sinusoidal_pe(max_len, d_model))

    def forward(self, src, src_mask):
        batch_size, src_len = src.shape
        pos = torch.arange(0, src_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        src = self.dropout((self.tok_embedding(src) * self.scale.to(src.device)) + self.pos_table[pos])
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=512):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))

        self.register_buffer('pos_table', _build_sinusoidal_pe(max_len, d_model))

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        batch_size, tgt_len = tgt.shape
        pos = torch.arange(0, tgt_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1)
        tgt = self.dropout((self.tok_embedding(tgt) * self.scale.to(tgt.device)) + self.pos_table[pos])
        for layer in self.layers:
            tgt, attention = layer(tgt, memory, tgt_mask, memory_mask)
        output = self.fc_out(tgt)
        return output, attention


# Transformer模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encoder(src, src_mask)
        output, attention = self.decoder(tgt, memory, tgt_mask, src_mask)
        return output, attention