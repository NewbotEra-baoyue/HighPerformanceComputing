"""训练循环:lr warmup+cosine、梯度裁剪、AMP、验证 loss、断点续训"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIGS
from data import Vocab, load_smiles, make_loader
from model import GPT

CKPT_DIR = Path(__file__).parent / "ckpt"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, cfg) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    p = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0
        )
        total += loss.item() * y.numel()
        count += y.numel()
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_quick", choices=list(CONFIGS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="从 ckpt/last.pt 继续训练")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖预设步数(冒烟测试用)")
    args = ap.parse_args()
    cfg = dict(CONFIGS[args.config])
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    device = pick_device(args.device)
    amp = cfg["amp"] and device.type == "cuda"
    torch.manual_seed(42)

    print(f"device={device}, amp={amp}, config={args.config}")
    vocab = Vocab.load()
    train_smiles = load_smiles("train", cfg["train_limit"])
    valid_smiles = load_smiles("valid", cfg["valid_limit"])
    print(f"train={len(train_smiles)}, valid={len(valid_smiles)}, vocab={vocab.size}")

    train_loader = make_loader(train_smiles, vocab, cfg)
    valid_loader = make_loader(valid_smiles, vocab, cfg, shuffle=False)

    model = GPT(
        vocab.size,
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    if cfg["compile"] and device.type == "cuda":
        model = torch.compile(model)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, best_val = 0, float("inf")
    if args.resume and (CKPT_DIR / "last.pt").exists():
        ck = torch.load(CKPT_DIR / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {step}, best_val={best_val:.4f}")

    CKPT_DIR.mkdir(exist_ok=True)
    model.train()
    t0 = time.time()
    loss_acc, cnt = 0.0, 0
    pbar = tqdm(total=cfg["max_steps"], initial=step, desc="train")

    while step < cfg["max_steps"]:
        for x, y in train_loader:
            if step >= cfg["max_steps"]:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=vocab.pad_id,
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()

            loss_acc += loss.item()
            cnt += 1
            step += 1
            pbar.update(1)
            if step % cfg["log_every"] == 0:
                pbar.set_postfix(loss=f"{loss_acc/cnt:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
                loss_acc, cnt = 0.0, 0

            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                val = evaluate(model, valid_loader, device, amp)
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {
                        "model": raw.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "best_val": min(best_val, val),
                        "config": args.config,
                    },
                    CKPT_DIR / "last.pt",
                )
                if val < best_val:
                    best_val = val
                    torch.save(
                        {"model": raw.state_dict(), "config": args.config},
                        CKPT_DIR / "best.pt",
                    )
                tqdm.write(
                    f"step {step}: val_loss={val:.4f}, best={best_val:.4f}, "
                    f"{(time.time()-t0)/step:.3f}s/step"
                )
    pbar.close()
    print(f"done. best val_loss={best_val:.4f}, ckpt in {CKPT_DIR}")


if __name__ == "__main__":
    main()
