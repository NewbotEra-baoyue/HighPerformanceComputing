#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MolecularGPT: 基于 Transformer 的自回归药物分子生成模型
用法：
  训练: python molecular_gpt.py --mode train --data_dir ./data --epochs 20
  生成: python molecular_gpt.py --mode generate --model_path ./checkpoints/best_model.pt --start_fragment "c1cc"

推理阶段已集成 KV Cache，大幅提升长序列生成速度。
"""

import os
import math
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import pickle

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------------------
# 数据处理
# ----------------------------------------------------------------------
SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def build_vocab(smiles_list, min_freq=1):
    """根据 SMILES 列表构建字符级词汇表"""
    all_chars = ''.join(smiles_list)
    char_counts = Counter(all_chars)
    if min_freq > 1:
        char_counts = {c: f for c, f in char_counts.items() if f >= min_freq}
    sorted_chars = [c for c, _ in char_counts.most_common()]
    vocab = SPECIAL_TOKENS + sorted_chars
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(vocab)

def encode_smiles(smiles, char2idx):
    tokens = [BOS_IDX] + [char2idx.get(c, UNK_IDX) for c in smiles] + [EOS_IDX]
    return tokens

def decode_tokens(indices, idx2char):
    chars = []
    for i in indices:
        if i in (PAD_IDX, BOS_IDX, EOS_IDX):
            continue
        chars.append(idx2char.get(i, ''))
    return ''.join(chars)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char2idx, max_len=128):
        self.smiles_list = smiles_list
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        tokens = encode_smiles(smi, self.char2idx)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [PAD_IDX] * (self.max_len - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

# ----------------------------------------------------------------------
# 模型定义（支持 KV Cache）
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x, offset=0):
        # x: (B, T, d_model)
        return x + self.pe[:, offset:offset + x.size(1), :]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # (B, nh, T, hs)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)     # (B, nh, T_total, hs)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            new_kv = (k, v)
        else:
            new_kv = None

        # 注意力分数 (q 的长度为 T，k/v 的长度为 T_total)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # 若 mask 尺寸与 att 不匹配（例如使用缓存时 T=1），只截取必要的部分
            # 此时 mask 应为 (1,1,1,T_total) 或 (1,1,T,T)，但为了简单，生成时我们传入 None
            att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.proj(out)
        return out, new_kv

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), mask, past_kv, use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv

class MolecularGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, ff_dim=2048,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享
        self.token_embed.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None, past_key_values=None, use_cache=False):
        """
        Args:
            x: (B, T) token indices
            mask: 因果掩码，形状 (1,1,T,T) 或 None (训练时必须提供)
            past_key_values: 列表，每个元素为 (k, v)，长度 n_layers
            use_cache: 是否返回缓存
        Returns:
            logits: (B, T, vocab_size)
            new_past: 新的缓存列表（若 use_cache=True）
        """
        # 计算位置偏移
        if past_key_values is not None:
            offset = past_key_values[0][0].size(2)   # (B, nh, T_past, hs)
        else:
            offset = 0

        x = self.token_embed(x)          # (B, T, d_model)
        x = self.pos_enc(x, offset=offset)

        new_past = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = block(x, mask, past_kv, use_cache)
            if use_cache:
                new_past.append(kv)

        x = self.ln_f(x)
        logits = self.head(x)            # (B, T, vocab_size)
        return logits, new_past if use_cache else None

def create_causal_mask(seq_len, device):
    """创建下三角因果掩码 (1,1,T,T)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    return mask

# ----------------------------------------------------------------------
# 训练与验证
# ----------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Training")
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        mask = create_causal_mask(T, device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(x, mask, use_cache=False)  # 训练时不使用缓存
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x, mask, use_cache=False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            mask = create_causal_mask(T, device)
            logits, _ = model(x, mask, use_cache=False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def generate(model, start_fragment, char2idx, idx2char, device,
             max_new_tokens=100, temperature=1.0):
    """
    使用 KV Cache 自回归生成 SMILES。
    """
    model.eval()

    # 编码起始片段（含 BOS）
    init_tokens = [BOS_IDX] + [char2idx.get(c, UNK_IDX) for c in start_fragment]
    init_ids = torch.tensor(init_tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        # ---- 第一步：处理初始序列，得到缓存和最后一个 token 的 logits ----
        mask = create_causal_mask(init_ids.size(1), device)
        logits, past_key_values = model(init_ids, mask=mask, use_cache=True)
        logits_last = logits[:, -1, :]   # (1, vocab)

        # 采样第一个新 token
        if temperature != 1.0:
            logits_last = logits_last / temperature
        probs = torch.softmax(logits_last, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated = init_tokens + [next_token]

        # 如果第一个 token 就是 EOS，直接返回（去掉特殊标记）
        if next_token == EOS_IDX:
            return decode_tokens(generated[1:], idx2char)

        # ---- 后续逐 token 生成 ----
        for _ in range(max_new_tokens - 1):
            # 输入只有最后一个 token (shape: 1,1)
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            # 使用缓存，无需 mask（单 token 无因果限制）
            logits, past_key_values = model(
                last_token,
                past_key_values=past_key_values,
                use_cache=True,
                mask=None
            )
            logits_last = logits[:, -1, :]   # (1, vocab)
            if temperature != 1.0:
                logits_last = logits_last / temperature
            probs = torch.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == EOS_IDX:
                break
            generated.append(next_token)

    # 解码（去除 BOS 和 EOS）
    decoded = decode_tokens(generated[1:], idx2char)
    return decoded

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MolecularGPT: 药物分子生成')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'],
                        help='运行模式：训练或生成')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='MOSES 数据集目录（包含 train.csv 和 test.csv）')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                        help='模型保存/加载路径')
    parser.add_argument('--start_fragment', type=str, default='c1cc',
                        help='生成模式下的起始片段（SMILES 子串）')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='生成的最大 token 数')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度')
    parser.add_argument('--num_samples', type=int, default=1,
                    help='生成分子的数量（仅 generate 模式）')
    parser.add_argument('--output_file', type=str, default='generated_mols.txt',
                    help='保存生成分子的文件路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--ff_dim', type=int, default=2048)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='限制训练样本数量（用于快速训练），默认使用全部数据')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if args.mode == 'train':
        # 加载数据
        train_path = os.path.join(args.data_dir, 'train.csv')
        valid_path = os.path.join(args.data_dir, 'test.csv')
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            raise FileNotFoundError("未找到 train.csv 或 test.csv，请检查 --data_dir")
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        train_smiles = train_df['SMILES'].tolist()
        valid_smiles = valid_df['SMILES'].tolist()
        
        if args.max_train_samples is not None and args.max_train_samples > 0:
            if len(train_smiles) > args.max_train_samples:
                indices = np.random.choice(len(train_smiles), args.max_train_samples, replace=False)
                train_smiles = [train_smiles[i] for i in indices]
            print(f"随机抽取 {args.max_train_samples} 条训练样本")
        
        print(f"训练集大小: {len(train_smiles)}，验证集大小: {len(valid_smiles)}")

        # 构建词汇表
        char2idx, idx2char, vocab_size = build_vocab(train_smiles, min_freq=1)
        print(f"词汇表大小: {vocab_size}")

        # 保存词汇表（用于推理）
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path + '.vocab', 'wb') as f:
            pickle.dump({'char2idx': char2idx, 'idx2char': idx2char, 'vocab_size': vocab_size}, f)

        # 创建数据集和数据加载器
        train_dataset = SMILESDataset(train_smiles, char2idx, max_len=args.max_len)
        valid_dataset = SMILESDataset(valid_smiles, char2idx, max_len=args.max_len)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # 初始化模型
        model = MolecularGPT(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_dim=args.ff_dim,
            max_len=args.max_len,
            dropout=args.dropout
        ).to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == 'cuda') else None

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            print(f"\n===== Epoch {epoch}/{args.epochs} =====")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_ppl = evaluate(model, valid_loader, criterion, device)
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid Perplexity: {val_ppl:.2f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }, args.model_path)
                print(f"模型已保存至 {args.model_path}")

        print("训练完成！")

    elif args.mode == 'generate':
        # 加载词汇表
        vocab_path = args.model_path + '.vocab'
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"未找到词汇表文件 {vocab_path}")
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        char2idx = vocab_data['char2idx']
        idx2char = vocab_data['idx2char']
        vocab_size = vocab_data['vocab_size']

        # 加载模型
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model_args = checkpoint['args']
        model = MolecularGPT(
            vocab_size=vocab_size,
            d_model=model_args.d_model,
            n_heads=model_args.n_heads,
            n_layers=model_args.n_layers,
            ff_dim=model_args.ff_dim,
            max_len=model_args.max_len,
            dropout=model_args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功")

        start_time = time.time()

        generated_smiles = generate(
            model, args.start_fragment, char2idx, idx2char, device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        elapsed = time.time() - start_time
        print(f"起始片段: {args.start_fragment}")
        print(f"生成分子: {generated_smiles}")
        print(f"推理耗时: {elapsed:.4f} 秒")
        
        
    #     start_time = time.time()
    #     generated_list = []
    #     for i in range(args.num_samples):
    #         smi = generate(
    #         model, args.start_fragment, char2idx, idx2char, device,
    #         max_new_tokens=args.max_new_tokens, temperature=args.temperature
    #         )
    #     generated_list.append(smi)
    #     elapsed = time.time() - start_time
    #     print(f"推理耗时: {elapsed:.4f} 秒")
    #     print(f"[{i+1}/{args.num_samples}] {smi}")
    # # 保存到文件
    # with open(args.output_file, 'w') as f:
    #     f.write('\n'.join(generated_list))
    # print(f"所有生成结果已保存至 {args.output_file}")

        # 可选 RDKit 有效性检查
        # try:
        #     from rdkit import Chem
        #     mol = Chem.MolFromSmiles(generated_smiles)
        #     if mol is not None:
        #         print("RDKit 检查: 有效分子")
        #     else:
        #         print("RDKit 检查: 无效 SMILES")
        # except ImportError:
        #     print("未安装 RDKit，跳过有效性检查")

if __name__ == '__main__':
    main()