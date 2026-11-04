"""Bounded SYNTHETIC memory-task PPO diagnostic. Not Civ6 training.

An initial +/- cue predicts a binary terminal choice three decisions later.
The reward and state here are artificial and must never qualify the host.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .encoding import MAP_CHANNELS, GLOBAL_FEATURES, ENTITY_FEATURES, ACTION_FEATURES
from .model import PolicyConfig, RecurrentPolicy, collate, checkpoint_payload
from .ppo import generalized_advantages, update
from .monitor import append_metric


def frame(cue):
    actions = [[0.] * len(ACTION_FEATURES) for _ in range(2)]
    actions[0][0] = 1.; actions[1][1] = 1.
    return {"map": [[[0.] * 6 for _ in range(6)] for _ in MAP_CHANNELS],
            "global": {"values": [float(cue)] + [0.] * (len(GLOBAL_FEATURES) - 1), "mask": [1.] * len(GLOBAL_FEATURES)},
            "entities": [], "entity_mask": [], "candidates": actions,
            "candidate_ids": ["synthetic-left", "synthetic-right"], "action_mask": [True, True]}


def sync(device):
    if device == "mps": torch.mps.synchronize()
    elif device.startswith("cuda"): torch.cuda.synchronize()


def run(destination, *, device="cpu", updates=12, batch_size=16, seed=1, time_limit=120):
    if not 1 <= updates <= 64 or not 2 <= batch_size <= 64 or not 1 <= time_limit <= 180:
        raise ValueError("diagnostic budget exceeded")
    torch.manual_seed(seed); torch.set_num_threads(4)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    config = PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES), len(ENTITY_FEATURES), len(ACTION_FEATURES))
    manifest = {"schema": "civ6rl.learning_run.v1", "run_id": destination.name, "mode": "diagnostic",
                "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "training_eligible": False,
                "limitations": ["Tâche de mémoire synthétique : aucune partie Civ6, aucun taux de victoire Civ6.",
                                "Essai borné de CNN/LSTM/PPO et du moniteur ; ne qualifie pas l'hôte natif."],
                "device": device, "seed": seed, "batch_size": batch_size, "config": asdict(config), "torch": torch.__version__}
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        model = RecurrentPolicy(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        steps = 0
        for iteration in range(updates):
            if time.perf_counter() - start > time_limit: break
            cues = torch.randint(0, 2, (batch_size,)).tolist()
            encoding_start = time.perf_counter()
            batches = [collate([frame(2 * cue - 1 if t == 0 else 0) for cue in cues], device) for t in range(4)]
            sync(device); encoding_seconds = time.perf_counter() - encoding_start
            infer_start = time.perf_counter()
            actions = []; values = []; logps = []; state = None
            with torch.no_grad():
                for batch in batches:
                    dist, value, state = model(batch, state)
                    action = dist.sample()
                    actions.append(action); values.append(value); logps.append(dist.log_prob(action))
            sync(device); inference_seconds = time.perf_counter() - infer_start
            actions, values, logps = torch.stack(actions), torch.stack(values), torch.stack(logps)
            rewards = torch.zeros_like(values)
            rewards[-1] = actions[-1].eq(torch.tensor(cues, device=device)).float() * 2 - 1
            terminated = torch.zeros_like(rewards, dtype=torch.bool); terminated[-1] = True
            next_values = torch.cat((values[1:], torch.zeros_like(values[:1])))
            advantages, returns = generalized_advantages(rewards, values, next_values, terminated,
                torch.zeros_like(terminated), torch.ones_like(values))
            update_start = time.perf_counter()
            metrics = update(model, optimizer, batches, actions, logps, advantages, returns)
            sync(device); update_seconds = time.perf_counter() - update_start
            steps += 4 * batch_size
            wall = time.perf_counter() - start
            metrics.update({"reward": rewards.sum(0).mean().item(), "encoding_seconds": encoding_seconds,
                            "inference_seconds": inference_seconds, "update_seconds": update_seconds,
                            "steps_per_second": steps / wall, "synthetic_success_rate": (rewards[-1] > 0).float().mean().item()})
            append_metric(destination, steps, wall, "synthetic_memory_diagnostic", metrics)
        changed = any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        if not changed or steps == 0: raise RuntimeError("no parameter update observed")
        torch.save(checkpoint_payload(model), destination / "policy.pt")
        manifest.update(status="completed", samples=steps, weights_changed=changed,
                        wall_seconds=time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters()))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc)[:1000])
        raise
    finally:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True); p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--updates", type=int, default=12); p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(run(args.output, device=args.device, updates=args.updates, batch_size=args.batch_size)))

if __name__ == "__main__": main()
