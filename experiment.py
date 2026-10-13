"""Train standard DQN, freeze checkpoints, then evaluate held-out workloads."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import gymnasium
from routing_env import RoutingEnv, rule_policy, workload, evaluate, CATEGORIES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

@dataclass
class Config:
    episodes: int = 10000
    lr: float = .001
    gamma: float = .95
    batch_size: int = 64
    buffer_size: int = 10000
    warmup: int = 500
    target_interval: int = 250
    epsilon_decay: int = 8000
    validation_interval: int = 500
    validation_seed: int = 41000
    test_seed: int = 51000
    validation_tasks: int = 400
    test_tasks: int = 1000

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)

class Greedy:
    def __init__(self, model):
        self.model = model

    def __call__(self, state):
        with torch.no_grad():
            return self.model(torch.from_numpy(state)).argmax().item()

class Replay:
    def __init__(self, size):
        self.s = np.zeros((size, 7), np.float32)
        self.ns = np.zeros_like(self.s)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.size = size
        self.count = 0

    def add(self, s, a, r, ns, d):
        i = self.count % self.size
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, d
        self.count += 1

    def sample(self, rng, n):
        idx = rng.choice(min(self.count, self.size), n, replace=False)
        return [torch.from_numpy(x[idx]) for x in [self.s, self.a, self.r, self.ns, self.d]]

def metrics(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in ['success', 'total_reward', 'calls', 'cost', 'latency', 'decisions']}

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def train(seed, config, name, deterministic=False):
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = RoutingEnv(deterministic=deterministic)
    env.reset(seed=seed)
    model, target = DQN(), DQN()
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    replay = Replay(config.buffer_size)
    validation = workload(config.validation_seed, config.validation_tasks)
    if deterministic:
        validation['draws'].fill(0)
    logs, validations = [], []
    best_return, best_episode, updates = -float('inf'), 0, 0
    started = time.perf_counter()
    for episode in range(1, config.episodes + 1):
        s, _ = env.reset()
        epsilon = max(.05, 1 - .95 * (episode - 1) / config.epsilon_decay)
        losses = []
        while True:
            a = int(rng.integers(4)) if rng.random() < epsilon else Greedy(model)(s)
            ns, r, done, _, info = env.step(a)
            replay.add(s, a, r, ns, done)
            s = ns
            if replay.count >= config.warmup:
                bs, ba, br, bns, bd = replay.sample(rng, config.batch_size)
                q = model(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    y = br + config.gamma * (1 - bd) * target(bns).max(1).values
                loss = nn.functional.smooth_l1_loss(q, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                losses.append(loss.item())
                if updates % config.target_interval == 0:
                    target.load_state_dict(model.state_dict())
            if done:
                break
        logs.append(dict(episode=episode, epsilon=epsilon, reward=info['total_reward'], success=int(info['success']),
                         calls=info['calls'], loss=float(np.mean(losses)) if losses else ''))
        if episode % config.validation_interval == 0:
            m = metrics(evaluate(Greedy(model), validation))
            validations.append(dict(episode=episode, **m))
            if m['total_reward'] > best_return:
                best_return, best_episode = m['total_reward'], episode
                torch.save(dict(state_dict=model.state_dict(), seed=seed, config=asdict(config), episode=episode), OUT / 'checkpoints' / f'{name}.pt')
            print(f'{name} episode={episode} validation return={m["total_reward"]:.3f} success={m["success"]:.3f}', flush=True)
    elapsed = time.perf_counter() - started
    write_csv(OUT / f'{name}_training.csv', logs)
    write_csv(OUT / f'{name}_validation.csv', validations)
    near = [x['episode'] for x in validations if best_return > 0 and x['total_reward'] >= .95 * best_return]
    return dict(name=name, seed=seed, lr=config.lr, best_validation_return=best_return, checkpoint_episode=best_episode,
                training_seconds=elapsed, episodes_to_95pct_best=min(near) if near else None, updates=updates)

def load_policy(name):
    checkpoint = torch.load(OUT / 'checkpoints' / f'{name}.pt', weights_only=True)
    model = DQN()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return Greedy(model)

def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    config = Config()
    (OUT / 'experiment_config.json').write_text(json.dumps(asdict(config), indent=2))
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, gymnasium=gymnasium.__version__, device='CPU', threads=1)
    (OUT / 'versions.json').write_text(json.dumps(versions, indent=2))
    smoke_config = Config(episodes=2000, epsilon_decay=1500)
    smoke = train(7, smoke_config, 'deterministic_smoke', deterministic=True)
    smoke_tasks = workload(42000, 120)
    smoke_tasks['draws'].fill(0)
    smoke_result = metrics(evaluate(load_policy('deterministic_smoke'), smoke_tasks))
    if smoke_result['success'] < .99:
        raise RuntimeError('DQN deterministic smoke run did not learn all categories')
    (OUT / 'deterministic_smoke.json').write_text(json.dumps(dict(run=smoke, evaluation=smoke_result), indent=2))
    # Both full pilots use the same training seed and fixed validation set.
    pilots = []
    for lr in [.001, .0003]:
        config.lr = lr
        pilots.append(train(71, config, f'pilot_{lr}'))
    selected = max(pilots, key=lambda x: x['best_validation_return'])['lr']
    config.lr = selected
    (OUT / 'selected_config.json').write_text(json.dumps(asdict(config), indent=2))
    runs = [train(seed, config, f'dqn_{seed}') for seed in [11, 22, 33, 44, 55]]
    # No test evaluation occurs until all configurations and checkpoints are frozen.
    tasks = workload(config.test_seed, config.test_tasks)
    np.savez_compressed(OUT / 'test_workload.npz', **tasks)
    np.savez_compressed(OUT / 'validation_workload.npz', **workload(config.validation_seed, config.validation_tasks))
    rows, summaries, traces = [], [], []
    for shift, scenario in [(0.0, 'standard'), (-.1, 'reduced_reliability')]:
        for seed in [11, 22, 33, 44, 55]:
            random_rng = np.random.default_rng(61000 + seed)
            policies = {'DQN': load_policy(f'dqn_{seed}'), 'Rule': rule_policy, 'Random': lambda s: random_rng.integers(4)}
            for label, policy in policies.items():
                samples = evaluate(policy, tasks, reliability_shift=shift, traces=label == 'DQN' and seed == 11 and shift == 0)
                for sample in samples:
                    if 'trace' in sample:
                        traces.append(sample.copy())
                    rows.append(dict(scenario=scenario, policy=label, seed=seed, **{k:v for k,v in sample.items() if k != 'trace'}))
                summaries.append(dict(scenario=scenario, policy=label, seed=seed, **metrics(samples)))
    write_csv(OUT / 'episode_results.csv', rows)
    write_csv(OUT / 'run_metrics.csv', summaries)
    grouped = []
    for scenario in ['standard', 'reduced_reliability']:
        for policy in ['DQN', 'Rule', 'Random']:
            subset = [r for r in summaries if r['scenario'] == scenario and r['policy'] == policy]
            group = dict(scenario=scenario, policy=policy)
            for key in ['success', 'total_reward', 'calls', 'cost', 'latency', 'decisions']:
                group[key + '_mean'] = float(np.mean([r[key] for r in subset]))
                group[key + '_sd'] = float(np.std([r[key] for r in subset], ddof=1))
            grouped.append(group)
    write_csv(OUT / 'aggregate_metrics.csv', grouped)
    category_results = []
    for scenario in ['standard', 'reduced_reliability']:
        for policy in ['DQN', 'Rule', 'Random']:
            for category in range(4):
                per_seed = [metrics([r for r in rows if r['scenario'] == scenario and r['policy'] == policy and r['category'] == category and r['seed'] == seed]) for seed in [11,22,33,44,55]]
                category_results.append(dict(scenario=scenario, policy=policy, category=CATEGORIES[category],
                    success_mean=float(np.mean([r['success'] for r in per_seed])), success_sd=float(np.std([r['success'] for r in per_seed], ddof=1)),
                    calls_mean=float(np.mean([r['calls'] for r in per_seed]))))
    write_csv(OUT / 'category_metrics.csv', category_results)
    dqn, rule, random_result = [next(r for r in grouped if r['scenario'] == 'standard' and r['policy'] == p) for p in ['DQN','Rule','Random']]
    targets = dict(success_gain_pp=100*(dqn['success_mean']-random_result['success_mean']),
                   rule_success_gap_pp=100*(rule['success_mean']-dqn['success_mean']),
                   cost_ratio=dqn['cost_mean']/rule['cost_mean'])
    targets['met'] = dict(random_gain=targets['success_gain_pp'] >= 20, rule_gap=abs(targets['rule_success_gap_pp']) <= 5, cost=targets['cost_ratio'] <= 1.1)
    report = dict(pilots=pilots, selected_lr=selected, runs=runs, aggregates=grouped, targets=targets,
                  note='SD across five trained policies on one shared 1000-task test set. Rule policy is deterministic and repeated on the same workload. This is not uncertainty across independent test workloads.')
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    (OUT / 'demo_traces.json').write_text(json.dumps(traces, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    run()
