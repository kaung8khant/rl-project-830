"""Execute the DQN router and print routing to simulated subagents."""
import argparse
import json
from time import perf_counter
import numpy as np
import torch
from experiment import ROOT, load_policy
from routing_env import RoutingEnv, CATEGORIES, ACTIONS


def yes(value):
    return 'YES' if value else 'NO'


def run_case(policy, tasks, task_id, shift, show_q=False):
    env = RoutingEnv(reliability_shift=shift)
    state, _ = env.reset(seed=0, options={k: tasks[k][task_id] for k in tasks.files})
    print(f'Test task {task_id}: {CATEGORIES[env.category]}', flush=True)
    print(f'Complexity: {["Low", "Medium", "High"][env.complexity]} | Decision budget: 3', flush=True)
    print(f'Requires data: {yes(state[1])} | Requires analysis: {yes(state[2])}', flush=True)
    while True:
        print(f'\nDecision {env.decisions + 1}/3', flush=True)
        print(f'  Current state: data available={yes(state[3])}, analysis complete={yes(state[4])}', flush=True)
        with torch.no_grad():
            q = policy.model(torch.from_numpy(state))
            action = int(q.argmax().item())
        print(f'  DQN selected route: {ACTIONS[action]}', flush=True)
        if show_q:
            print('  Q [Direct, SQL subagent, Analysis subagent, Combined]: ' + ', '.join(f'{v:.2f}' for v in q.tolist()), flush=True)
        state, reward, done, _, info = env.step(action)
        for event in info['events']:
            print(f'  Environment: {event}', flush=True)
        print(f'  Updated state: data available={yes(state[3])}, analysis complete={yes(state[4])}', flush=True)
        print(f'  Reward: {reward:+.1f} | Total return: {info["total_reward"]:+.1f}', flush=True)
        print(f'  Subagent calls: {info["calls"]} | Simulated latency units: {info["latency"]:.0f}', flush=True)
        if done:
            print('\nResult: ' + ('TASK COMPLETED' if info['success'] else 'FAILED: DECISION BUDGET EXHAUSTED'), flush=True)
            return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, help='Run one test task instead of all six examples')
    parser.add_argument('--seed', type=int, default=11, choices=[11, 22, 33, 44, 55])
    parser.add_argument('--shift', type=float, default=0, help='Operation reliability shift, e.g. -0.1')
    parser.add_argument('--show-q', action='store_true', help='Print predicted action values')
    parser.add_argument('--presentation', action='store_true', help='Run six examples (also the default)')
    args = parser.parse_args()
    if args.task is not None and args.presentation:
        parser.error('Choose either --task or --presentation')
    started = perf_counter()
    print('DQN SUBAGENT ROUTING: ACTUAL CODE EXECUTION', flush=True)
    print('Router: trained DQN. Subagents: simulated SQL and analysis roles.', flush=True)
    print('SQL subagent handles retrieval. Analysis subagent handles interpretation.', flush=True)
    print('Exploration OFF. Subagent outcomes are simulated; no live LLM calls.', flush=True)
    print(f'Loading trained checkpoint: results/checkpoints/dqn_{args.seed}.pt', flush=True)
    policy = load_policy(f'dqn_{args.seed}')
    print('Checkpoint loaded successfully.', flush=True)
    tasks = np.load(ROOT / 'results/test_workload.npz')
    if args.task is not None:
        if not 0 <= args.task < len(tasks['category']):
            parser.error('Task index must be 0 to 999')
        ids = [args.task]
    else:
        selected = json.loads((ROOT / 'results/selected_demo_episodes.json').read_text())
        ids = [case['task_id'] for case in selected]
    summary = []
    for index, task_id in enumerate(ids, 1):
        print('\n' + '=' * 72, flush=True)
        print(f'CASE {index}/{len(ids)}', flush=True)
        print('=' * 72, flush=True)
        info = run_case(policy, tasks, task_id, args.shift, args.show_q)
        summary.append((task_id, info))
    print('\n' + '=' * 72)
    print('EXECUTION SUMMARY')
    for task_id, info in summary:
        print(f'Task {task_id:3d}: {"SUCCESS" if info["success"] else "FAILURE":7s} | Calls {info["calls"]} | Return {info["total_reward"]:+.1f}')
    print(f'Actual checkpoint loading and demo execution time: {perf_counter() - started:.4f} seconds')
    print('Selected examples illustrate behavior; aggregate test metrics measure performance.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExecution interrupted.')
