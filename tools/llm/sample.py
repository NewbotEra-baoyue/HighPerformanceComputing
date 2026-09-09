"""采样生成 SMILES,输出到 outputs/samples.csv 并统计 validity/uniqueness"""

import argparse
import csv
import re
from pathlib import Path

import torch

from config import CONFIGS
from data import Vocab
from model import GPT

OUT_DIR = Path(__file__).parent / "outputs"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def check_smiles(smi: str) -> bool:
    """轻量检查:括号配平、环闭合数字成对出现"""
    if not smi or len(smi) < 2:
        return False
    if smi.count("(") != smi.count(")") or smi.count("[") != smi.count("]"):
        return False
    for d in "123456789":
        if smi.count(d) % 2 != 0:
            return False
    return bool(re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$%\\/\.:]+", smi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--config", default=None, help="默认读取 checkpoint 中的 config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = CONFIGS[args.config or ck.get("config", "cpu_quick")]

    vocab = Vocab.load()
    model = GPT(
        vocab.size,
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "samples.csv"
    gen, seen = [], set()
    temp = 0.0 if args.greedy else args.temperature

    with torch.no_grad():
        for _ in range(0, args.n, args.batch):
            b = min(args.batch, args.n - len(gen))
            idx = torch.full((b, 1), vocab.bos_id, dtype=torch.long, device=device)
            idx = model.generate(
                idx,
                args.max_len,
                temperature=temp,
                top_k=args.top_k,
                eos_id=vocab.eos_id,
            )
            gen.extend(vocab.decode(ids.tolist()) for ids in idx)

    valid = [s for s in gen if check_smiles(s)]
    seen = set(valid)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SMILES"])
        w.writerows([[s] for s in valid])

    print(f"generated {len(gen)}, valid {len(valid)} ({len(valid)/max(len(gen),1):.1%}),")
    print(f"unique {len(seen)} ({len(seen)/max(len(valid),1):.1%}), saved -> {out_path}")
    print("examples:")
    for s in gen[:10]:
        print(" ", s)


if __name__ == "__main__":
    main()
