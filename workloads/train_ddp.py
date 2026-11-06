import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from grp_runtime.agent import RuntimeHook


def make_batch(step: int, n: int, dim: int):
    g = torch.Generator().manual_seed(10_000 + step)
    x = torch.randn(n, dim, generator=g)
    y = torch.tanh(x @ torch.linspace(-1, 1, dim)).unsqueeze(1) + 0.01 * torch.randn(n, 1, generator=g)
    return x, y


EVAL_SEED = 999_999


def make_eval(n: int, dim: int):
    return make_batch(EVAL_SEED - 10_000, n, dim)


def runtime_config() -> dict | None:
    raw = os.environ.get("AI_FACTORY_RUNTIME_CONFIG")
    return json.loads(raw) if raw else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--global-batch", type=int, default=1024)
    ap.add_argument("--per-rank-batch", type=int, default=512)
    ap.add_argument("--batch-mode", choices=["global", "per-rank"], default="global")
    ap.add_argument("--loss-log", default=None, help="per-step training loss CSV (rank 0)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate loss on fixed eval data every N steps (0=off)")
    ap.add_argument("--eval-size", type=int, default=4096)
    ap.add_argument("--eval-log", default=None, help="eval loss CSV (rank 0)")
    ap.add_argument("--save-final", default=None, help="path to save final weights when max-steps is reached (rank 0)")
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    ap.add_argument("--step-sleep", type=float, default=0.0, help="(demo) sleep seconds per step - simulates long iterations")
    ap.add_argument("--measure-warmup", type=int, default=None, help="report throughput and peak memory after this many warmup steps")
    a = ap.parse_args()

    backend = os.environ.get("GRP_DIST_BACKEND", "gloo")
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if backend == "nccl":
        dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    dim = 512
    model = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
                          nn.Linear(2048, 1)).to(dev)
    if a.strategy == "fsdp":
        from torch.distributed.fsdp import fully_shard
        for m in model:
            if isinstance(m, nn.Linear):
                fully_shard(m)
        fully_shard(model)
    else:
        model = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    eval_model = model.module if a.strategy == "ddp" else model

    hook = RuntimeHook(device=dev)
    start = hook.restore(model, opt)

    cfg = runtime_config()
    if cfg:
        global_n, per_rank = cfg["effectiveBatch"], cfg["effectiveBatch"] // world
        micro, accum = cfg["batch"]["microBatch"], cfg["batch"]["gradientAccumulation"]
        if micro * accum * world != global_n:
            raise SystemExit(f"runtime config {cfg['batch']} x world {world} != effectiveBatch {global_n}")
    else:
        per_rank = a.global_batch // world if a.batch_mode == "global" else a.per_rank_batch
        global_n, micro, accum = per_rank * world, per_rank, 1
    applied = {"effectiveBatch": global_n, "microBatch": micro, "gradientAccumulation": accum}
    log = open(a.loss_log, "a") if (a.loss_log and rank == 0) else None
    elog = open(a.eval_log, "a") if (a.eval_log and rank == 0) else None
    evaluating = bool(a.eval_every and a.eval_log and (rank == 0 or a.strategy == "fsdp"))
    if evaluating:
        ex, ey = (t.to(dev) for t in make_eval(a.eval_size, dim))

    step = start
    t0, first_loss = None, None
    for step in range(start, a.max_steps):
        if a.measure_warmup is not None and step == start + a.measure_warmup:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
        x, y = make_batch(step, global_n, dim)
        x, y = x[rank * per_rank:(rank + 1) * per_rank].to(dev), y[rank * per_rank:(rank + 1) * per_rank].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i in range(accum):
            sync = nullcontext() if i == accum - 1 or a.strategy != "ddp" else model.no_sync()
            with sync:
                part = nn.functional.mse_loss(model(x[i * micro:(i + 1) * micro]), y[i * micro:(i + 1) * micro]) / accum
                part.backward()
            loss = loss + part.detach()
        opt.step()

        lt = loss.detach().clone() if backend == "nccl" else loss.detach().cpu()
        dist.all_reduce(lt)
        gloss = lt.item() / world
        if first_loss is None:
            first_loss = gloss
        if log:
            log.write(f"{step + 1},{world},{gloss}\n")
            log.flush()
        if evaluating and (step + 1) % a.eval_every == 0:
            with torch.no_grad():
                eloss = nn.functional.mse_loss(eval_model(ex), ey).item()
            if elog:
                elog.write(f"{step + 1},{world},{eloss}\n")
                elog.flush()
        if a.step_sleep:
            time.sleep(a.step_sleep)

        if hook.step_end(step + 1, model, opt, loss=gloss, perRankBatch=per_rank,
                         globalBatch=global_n, appliedSettings=applied):
            break
    else:
        extra = {}
        if t0 is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elapsed = time.perf_counter() - t0
            measured = a.max_steps - start - a.measure_warmup
            peak = torch.tensor([torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else 0],
                                dtype=torch.float64, device=dev if backend == "nccl" else "cpu")
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            extra["measurement"] = {"throughput": measured * global_n / elapsed, "unit": "samples/s",
                                    "measuredSteps": measured, "seconds": elapsed,
                                    "peakMemoryBytesPerDevice": int(peak.item()),
                                    "firstLoss": first_loss, "lastLoss": gloss}
        hook.finish(step + 1, loss=gloss, appliedSettings=applied, **extra)
        if a.save_final:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if rank == 0:
                torch.save(sd, a.save_final)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
