"""MOSES 数据集下载、词表构建与 Dataset"""

import csv
import functools
import json
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocab.json"

BASE_URL = "https://github.com/molecularsets/moses/raw/master/data"
# MOSES 官方只发布 train/test,无 valid;验证集取 train 尾部留出
SPLITS = ("train", "test")
N_VALID_HELDOUT = 20_000

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def download(split: str) -> Path:
    if split == "valid":  # 复用 train.csv,见 load_smiles
        split = "train"
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        url = f"{BASE_URL}/{split}.csv"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_smiles(split: str, limit: int | None = None) -> list[str]:
    path = download(split)
    with open(path, newline="") as f:
        smiles = [row["SMILES"] for row in csv.DictReader(f)]
    if split == "valid":
        held = smiles[-N_VALID_HELDOUT:]  # train 尾部留出作验证集
        smiles = held[:limit] if limit else held
    elif limit is not None:
        smiles = smiles[:limit]
    return smiles


class Vocab:
    def __init__(self, chars: list[str]):
        self.itos = [PAD, BOS, EOS] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, smiles: str) -> list[int]:
        return [self.bos_id] + [self.stoi[c] for c in smiles] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            chars.append(self.itos[i])
        return "".join(chars)

    def save(self, path: Path = VOCAB_PATH):
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> "Vocab":
        v = cls([])
        v.itos = json.loads(path.read_text())
        v.stoi = {c: i for i, c in enumerate(v.itos)}
        return v


def build_vocab(smiles: list[str]) -> Vocab:
    return Vocab(sorted({c for s in smiles for c in s}))


class SmilesDataset(Dataset):
    """返回 (input_ids, target_ids),均已去掉/加上 BOS/EOS 对齐"""

    def __init__(self, smiles: list[str], vocab: Vocab, max_len: int):
        self.seqs = [vocab.encode(s)[: max_len + 1] for s in smiles]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        return torch.tensor(s[:-1]), torch.tensor(s[1:])


def collate(batch, pad_id: int):
    maxlen = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = maxlen - len(x)
        xs.append(torch.nn.functional.pad(x, (0, pad), value=pad_id))
        ys.append(torch.nn.functional.pad(y, (0, pad), value=pad_id))
    return torch.stack(xs), torch.stack(ys)


def collate_pad(pad_id: int, batch):
    return collate(batch, pad_id)


def make_loader(smiles, vocab, cfg, shuffle=True):
    ds = SmilesDataset(smiles, vocab, cfg["max_seq_len"])
    sampler = torch.utils.data.RandomSampler(ds) if shuffle else None

    def batch_sampler():
        # 按长度排序后切成近似等 token 数的 batch,再打乱顺序
        order = sorted(range(len(ds)), key=lambda i: len(ds.seqs[i]))
        batches, cur, cur_max = [], [], 0
        for i in order:
            l = len(ds.seqs[i])
            if cur and (len(cur) + 1) * max(cur_max, l) > cfg["batch_tokens"]:
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, l)
        if cur:
            batches.append(cur)
        if shuffle:
            import random

            random.shuffle(batches)
        return batches

    return torch.utils.data.DataLoader(
        ds,
        batch_sampler=batch_sampler(),
        collate_fn=functools.partial(collate_pad, vocab.pad_id),
        num_workers=2,
        persistent_workers=True,
    )


if __name__ == "__main__":
    for split in SPLITS:
        s = load_smiles(split)
        print(f"{split}: {len(s)} molecules, e.g. {s[0]}")
    vocab = build_vocab(load_smiles("train"))
    vocab.save()
    print(f"vocab size: {vocab.size}")
    print("".join(vocab.itos))
