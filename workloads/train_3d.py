import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn

from grp_runtime.agent import RuntimeHook


def make_batch(step: int, n: int, dim: int):
    g = torch.Generator().manual_seed(10_000 + step)
    x = torch.randn(n, dim, generator=g)
    y = torch.tanh(x @ torch.linspace(-1, 1, dim)).unsqueeze(1) + 0.01 * torch.randn(n, 1, generator=g)
    return x, y


COMM = torch.device("cpu")


def _all_reduce(t: torch.Tensor, group) -> torch.Tensor:
    c = t.detach().to(COMM, copy=True)
    dist.all_reduce(c, group=group)
    return c.to(t.device)


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, g):
        return _all_reduce(g, ctx.group), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce(x, group)

    @staticmethod
    def backward(ctx, g):
        return g, None


class Block(nn.Module):
    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, tp: int, tp_rank: int, group):
        super().__init__()
        shard = fc1.out_features // tp
        sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
        self.fc1 = nn.Linear(fc1.in_features, shard)
        self.fc2 = nn.Linear(shard, fc2.out_features, bias=False)
        self.bias = nn.Parameter(fc2.bias.detach().clone())
        with torch.no_grad():
            self.fc1.weight.copy_(fc1.weight[sl])
            self.fc1.bias.copy_(fc1.bias[sl])
            self.fc2.weight.copy_(fc2.weight[:, sl])
        self.tp, self.group = tp, group

    def forward(self, x):
        h = _CopyToTP.apply(x, self.group) if self.tp > 1 else x
        y = self.fc2(nn.functional.gelu(self.fc1(h)))
        y = _ReduceFromTP.apply(y, self.group) if self.tp > 1 else y
        return x + y + self.bias


SPLIT = {"fc1.weight": 0, "fc1.bias": 0, "fc2.weight": 1}


class Stage(nn.Module):
    def __init__(self, layers: list[nn.Module], names: list[str], tp: int = 1, tp_rank: int = 0, dp_rank: int = 0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.names, self.tp, self.tp_rank, self.dp_rank = names, tp, tp_rank, dp_rank

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def _params(self):
        for prefix, m in zip(self.names, self.layers):
            for n, p in m.named_parameters():
                yield f"{prefix}.{n}", SPLIT.get(n) if prefix.startswith("blocks.") else None, p

    def full_checkpoint(self, optimizer):
        pieces, step = {}, 0
        for key, dim, p in self._params():
            st = optimizer.state.get(p, {})
            step = max(step, int(st.get("step", 0)))
            if self.dp_rank or (dim is None and self.tp_rank):
                continue
            pieces[key] = (dim, self.tp_rank, p.detach().cpu(),
                           *(st[k].detach().cpu() if k in st else None for k in ("exp_avg", "exp_avg_sq")))
        rows = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object((pieces, step), rows, dst=0)
        if rows is None:
            return None
        parts: dict[str, list] = {}
        for got, _ in rows:
            for key, piece in got.items():
                parts.setdefault(key, []).append(piece)
        model, state = {}, {}
        for key, ps in parts.items():
            ps.sort(key=lambda x: x[1])
            dim = ps[0][0]
            join = (lambda i: ps[0][i]) if dim is None else                 (lambda i: None if ps[0][i] is None else torch.cat([x[i] for x in ps], dim))
            model[key] = join(2)
            state[key] = {"exp_avg": join(3), "exp_avg_sq": join(4)}
        return model, {"step": max(s for _, s in rows), "state": state}

    def load_full_checkpoint(self, ck, optimizer):
        model, opt = ck["model"], ck["optimizer"]
        for key, dim, p in self._params():
            def piece(t):
                if dim is None:
                    return t
                size = t.shape[dim] // self.tp
                return t.narrow(dim, self.tp_rank * size, size)
            with torch.no_grad():
                p.copy_(piece(model[key]).to(p.device))
            st = opt["state"].get(key) or {}
            if opt["step"] and st.get("exp_avg") is not None:
                optimizer.state[p] = {"step": torch.tensor(float(opt["step"])),
                                      "exp_avg": piece(st["exp_avg"]).to(p.device).clone(),
                                      "exp_avg_sq": piece(st["exp_avg_sq"]).to(p.device).clone()}


def full_model(dim: int, hidden: int, n_layers: int):
    torch.manual_seed(0)
    inp = nn.Linear(dim, hidden)
    blocks = [(nn.Linear(hidden, 4 * hidden), nn.Linear(4 * hidden, hidden)) for _ in range(n_layers)]
    for fc1, fc2 in blocks:
        nn.init.normal_(fc2.weight, std=0.02 / n_layers ** 0.5)
    head = nn.Linear(hidden, 1)
    return inp, blocks, head


class Tracker:
    def __init__(self, rank: int, layout: dict, params: dict):
        self.mlflow = self.run_id = None
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, layout)
        if rank != 0:
            return
        try:
            import mlflow
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "grp-parallel-train"))
            run = mlflow.start_run(run_name=os.environ.get("GRP_RUN_NAME") or os.environ.get("POD_NAME"),
                                   tags={"source": "grp-runtime", "pod": os.environ.get("POD_NAME", ""),
                                         "namespace": os.environ.get("POD_NAMESPACE", ""),
                                         "configMap": os.environ.get("GRP_CONFIGMAP", "")})
            mlflow.log_params(params)
            mlflow.log_dict({"ranks": rows}, "rank_layout.json")
            self.mlflow, self.run_id = mlflow, run.info.run_id
        except Exception as e:
            print(f"[train_3d] mlflow disabled: {e}", flush=True)

    def log(self, step: int, **metrics) -> None:
        if not self.mlflow:
            return
        try:
            self.mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"[train_3d] mlflow log failed at step {step}: {e}", flush=True)

    def end(self, status: str = "FINISHED") -> None:
        if self.mlflow:
            try:
                self.mlflow.end_run(status)
            except Exception as e:
                print(f"[train_3d] mlflow end failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--global-batch", type=int, default=1024)
    ap.add_argument("--micro-batches", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-log", default=None)
    ap.add_argument("--strategy", default="3d")
    a = ap.parse_args()

    global COMM
    dist.init_process_group(os.environ.get("GRP_DIST_BACKEND", "gloo"))
    if dist.get_backend() == "nccl":
        torch.cuda.set_device(0)
        COMM = torch.device("cuda", 0)
    rank, world = dist.get_rank(), dist.get_world_size()
    dp, tp, pp = (int(os.environ.get(k, d)) for k, d in (("GRP_DP", world), ("GRP_TP", 1), ("GRP_PP", 1)))
    if dp * tp * pp != world:
        raise SystemExit(f"DP×TP×PP={dp * tp * pp} != WORLD_SIZE={world}")
    if a.layers % pp or (4 * a.hidden) % tp or (a.global_batch // dp) % a.micro_batches:
        raise SystemExit("layers % PP, 4*hidden % TP and per-DP batch % micro-batches must be 0")
    groups = json.loads(os.environ.get("GRP_PROCESS_GROUPS") or "null")
    if groups is None:
        at = lambda p, d, t: (p * dp + d) * tp + t
        groups = {"tp": [[at(p, d, t) for t in range(tp)] for p in range(pp) for d in range(dp)],
                  "dp": [[at(p, d, t) for d in range(dp)] for p in range(pp) for t in range(tp)],
                  "pp": [[at(p, d, t) for p in range(pp)] for d in range(dp) for t in range(tp)]}
    mine = {}
    for kind in ("tp", "dp"):
        for ranks in groups[kind]:
            g = dist.new_group(ranks)
            if rank in ranks:
                mine[kind] = (g, ranks)
    tp_group, dp_group = mine["tp"][0], mine["dp"][0]
    stage = next(s for s in groups["pp"] if rank in s)
    tp_rank, dp_rank, pp_rank = mine["tp"][1].index(rank), mine["dp"][1].index(rank), stage.index(rank)
    prev_rank = stage[pp_rank - 1] if pp_rank > 0 else None
    next_rank = stage[pp_rank + 1] if pp_rank < pp - 1 else None
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dim = 512
    inp, blocks, head = full_model(dim, a.hidden, a.layers)
    per_stage = a.layers // pp
    lo, hi = pp_rank * per_stage, (pp_rank + 1) * per_stage
    layers = [Block(fc1, fc2, tp, tp_rank, tp_group) for fc1, fc2 in blocks[lo:hi]]
    names = [f"blocks.{i}" for i in range(lo, hi)]
    if pp_rank == 0:
        layers.insert(0, inp)
        names.insert(0, "inp")
    if pp_rank == pp - 1:
        layers.append(head)
        names.append("head")
    model = Stage(layers, names, tp, tp_rank, dp_rank).to(dev)
    del inp, blocks, head
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"[train_3d] rank {rank}/{world} dp={dp_rank}/{dp} tp={tp_rank}/{tp} pp={pp_rank}/{pp} "
          f"layers[{lo}:{hi}]{' +input' if pp_rank == 0 else ''}{' +head' if pp_rank == pp - 1 else ''} "
          f"params={n_params} device={dev_name} "
          f"visible={os.environ.get('NVIDIA_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    tracker = Tracker(rank, {"rank": rank, "dp": dp_rank, "tp": tp_rank, "pp": pp_rank, "layers": [lo, hi],
                             "input": pp_rank == 0, "head": pp_rank == pp - 1, "params": n_params,
                             "pod": os.environ.get("POD_NAME"), "device": dev_name},
                      {"dp": dp, "tp": tp, "pp": pp, "world": world, "hidden": a.hidden, "layers": a.layers,
                       "microBatches": a.micro_batches, "globalBatch": a.global_batch, "lr": a.lr,
                       "maxSteps": a.max_steps, "backend": dist.get_backend(), "device": dev_name})

    hook = RuntimeHook(device=dev)
    start = hook.restore(model, opt)
    if start:
        print(f"[train_3d] rank {rank} resumed at step {start} from {hook.resume}", flush=True)
    per_dp = a.global_batch // dp
    mb = per_dp // a.micro_batches
    log = open(a.loss_log, "a") if (a.loss_log and rank == 0) else None
    params = [p for p in model.parameters()]
    t0 = time.perf_counter()
    step = start
    for step in range(start, a.max_steps):
        x, y = make_batch(step, a.global_batch, dim)
        x, y = x[dp_rank * per_dp:(dp_rank + 1) * per_dp], y[dp_rank * per_dp:(dp_rank + 1) * per_dp]
        opt.zero_grad(set_to_none=True)
        saved, local_loss = [], 0.0
        for m in range(a.micro_batches):
            if pp_rank == 0:
                h_in = x[m * mb:(m + 1) * mb].to(dev)
            else:
                buf = torch.empty(mb, a.hidden, device=COMM)
                dist.recv(buf, src=prev_rank)
                h_in = buf.to(dev).requires_grad_()
            out = model(h_in)
            if next_rank is None:
                loss = nn.functional.mse_loss(out, y[m * mb:(m + 1) * mb].to(dev)) / a.micro_batches
                local_loss += loss.item()
            else:
                dist.send(out.detach().to(COMM), dst=next_rank)
                loss = None
            saved.append((h_in, out, loss))
        for h_in, out, loss in reversed(saved):
            if next_rank is None:
                loss.backward()
            else:
                g = torch.empty(mb, a.hidden, device=COMM)
                dist.recv(g, src=next_rank)
                out.backward(g.to(dev))
            if prev_rank is not None:
                dist.send(h_in.grad.detach().to(COMM), dst=prev_rank)
        if dp > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in params])
            flat = _all_reduce(flat, dp_group) / dp
            o = 0
            for p in params:
                n = p.numel()
                p.grad.copy_(flat[o:o + n].view_as(p))
                o += n
        opt.step()

        lt = torch.tensor([local_loss if next_rank is None else 0.0], dtype=torch.float64, device=COMM)
        dist.all_reduce(lt)
        gloss = lt.item() / (dp * tp)
        if log:
            log.write(f"{step + 1},{dp},{tp},{pp},{gloss}\n")
            log.flush()
        if rank == 0 and (step + 1) % 10 == 0:
            tracker.log(step + 1, loss=gloss, steps_per_sec=(step + 1 - start) / (time.perf_counter() - t0))
        if rank == 0 and ((step + 1) % 50 == 0 or step == start):
            print(f"[train_3d] step {step + 1} loss={gloss:.6f} {(step + 1 - start) / (time.perf_counter() - t0):.1f} step/s",
                  flush=True)
        if hook.step_end(step + 1, model, opt, loss=gloss, dp=dp, tp=tp, pp=pp,
                         microBatches=a.micro_batches, perDpBatch=per_dp, globalBatch=a.global_batch):
            tracker.end("KILLED")
            break
    else:
        tracker.log(step + 1, loss=gloss, steps_per_sec=(step + 1 - start) / (time.perf_counter() - t0))
        tracker.end()
        hook.finish(step + 1, loss=gloss, dp=dp, tp=tp, pp=pp,
                    **({"mlflowRunId": tracker.run_id} if tracker.run_id else {}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
