from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)


def batch(it: int, n: int, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(10_000 + it)
    x = torch.randn(n, dim, generator=g)
    w = torch.linspace(-1, 1, dim)
    y = torch.tanh(x @ w).unsqueeze(1) + 0.01 * torch.randn(n, 1, generator=g)
    return x.to(device), y.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dp", type=int, required=True)
    ap.add_argument("--global-batch", type=int, default=1024)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-iters", type=int, default=200_000)
    a = ap.parse_args()

    run = Path(a.run_dir)
    control, state = run / "control.json", run / "state.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    dim = 512
    model = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
                          nn.Linear(2048, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it, loss_v = 0, float("nan")
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it, loss_v = ck["iteration"], ck["loss"]
    start_it, t0 = it, time.perf_counter()
    info = {"pid": os.getpid(), "dp": a.dp, "globalBatch": a.global_batch,
            "microBatch": a.global_batch // a.dp,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "resumedFrom": a.resume, "resumedIteration": start_it}

    while it < a.max_iters:
        opt.zero_grad(set_to_none=True)
        x, y = batch(it, a.global_batch, dim, dev)
        total = 0.0
        for xs, ys in zip(x.chunk(a.dp), y.chunk(a.dp)):
            l = nn.functional.mse_loss(model(xs), ys) * (len(xs) / a.global_batch)
            l.backward()
            total += l.item()
        opt.step()
        it += 1
        loss_v = total

        cmd = None
        if control.exists():
            try:
                cmd = json.loads(control.read_text(encoding="utf-8")).get("request")
            except (json.JSONDecodeError, OSError):
                cmd = None
        if cmd == "safepoint":
            ckpt = run / f"ckpt-it{it}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iteration": it, "loss": loss_v}, ckpt)
            keep = {ckpt.resolve(), Path(a.resume).resolve() if a.resume else None}
            old = sorted(run.glob("ckpt-it*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[3:]
            for p in old:
                if p.resolve() not in keep:
                    p.unlink(missing_ok=True)
            control.unlink(missing_ok=True)
            write_json(state, {**info, "status": "safepoint", "iteration": it, "loss": loss_v,
                               "checkpoint": str(ckpt)})
            return
        if cmd == "stop":
            control.unlink(missing_ok=True)
            write_json(state, {**info, "status": "stopped", "iteration": it, "loss": loss_v})
            return
        if it % 10 == 0:
            el = time.perf_counter() - t0
            write_json(state, {**info, "status": "running", "iteration": it, "loss": loss_v,
                               "itPerSec": round((it - start_it) / el, 1) if el else None,
                               "gpuMemAllocMb": round(torch.cuda.memory_allocated() / 2**20, 1)
                               if dev.type == "cuda" else 0})
    write_json(state, {**info, "status": "finished", "iteration": it, "loss": loss_v})


if __name__ == "__main__":
    main()
