"""Route tasks to simulated subagent roles. No live LLM or database calls."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CATEGORIES = ['Direct', 'Retrieval', 'Analysis', 'Retrieval + analysis']
ACTIONS = ['Direct response', 'SQL subagent', 'Analysis subagent', 'SQL + Analysis subagents']
REQUIREMENTS = [(0, 0), (1, 0), (0, 1), (1, 1)]

class RoutingEnv(gym.Env):
    """The DQN selects a route; this environment simulates subagent outcomes.

    The SQL subagent represents retrieval and the analysis subagent represents
    interpretation. Their outputs are completion flags, not real query results
    or generated text. The combined route runs the two roles sequentially.
    """
    metadata = {'render_modes': ['ansi']}

    def __init__(self, reliability_shift=0.0, deterministic=False):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (7,), dtype=np.float32)
        self.reliability_shift = reliability_shift
        self.deterministic = deterministic
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.category = int(options.get('category', self.np_random.integers(4)))
        self.complexity = int(options.get('complexity', self.np_random.integers(3)))
        self.need_data, self.need_analysis = REQUIREMENTS[self.category]
        self.draws = np.asarray(options.get('draws', self.np_random.random((3, 3))))
        self.data = self.analysis = self.direct = False
        self.decisions = self.calls = 0
        self.latency = self.total_reward = 0.0
        self.outcome = 0.0  # no previous action=0, failure=.5, success=1
        self.done = False
        return self._state(), {}

    def _state(self):
        return np.array([self.complexity / 2, self.need_data, self.need_analysis,
                         self.data, self.analysis, self.outcome, self.decisions / 3], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset() required after termination')
        if not self.action_space.contains(action):
            raise ValueError('Action must be 0, 1, 2, or 3')
        p = 1.0 if self.deterministic else [0.95, 0.85, 0.75][self.complexity] + self.reliability_shift
        p = float(np.clip(p, 0, 1))
        calls, latency, events, operation_success = 0, 0.0, [], []
        if action == 0:
            latency = 1.0
            ok = self.category == 0 and self.draws[self.decisions, 0] < p
            self.direct = self.direct or ok
            operation_success.append(ok)
            events.append('Direct response succeeded' if ok else 'Direct response failed or requirements unmet')
        if action in (1, 3):
            calls += 1
            latency += 2
            ok = self.draws[self.decisions, 1] < p
            self.data = self.data or ok
            operation_success.append(ok)
            events.append('SQL subagent succeeded' if ok else 'SQL subagent failed')
        if action in (2, 3):
            calls += 1
            latency += 2
            available = not self.need_data or self.data
            ok = available and self.draws[self.decisions, 2] < p
            self.analysis = self.analysis or ok
            operation_success.append(ok)
            events.append('Analysis subagent succeeded' if ok else ('Analysis subagent failed' if available else 'Analysis subagent blocked: data missing'))
        self.decisions += 1
        success = self.direct if self.category == 0 else ((not self.need_data or self.data) and (not self.need_analysis or self.analysis))
        failure = self.decisions == 3 and not success
        self.done = bool(success or failure)
        self.outcome = 1.0 if all(operation_success) else 0.5
        reward = 10 * int(success) - 10 * int(failure) - calls - .1 * latency
        self.calls += calls
        self.latency += latency
        self.total_reward += reward
        info = dict(success=bool(success), calls=self.calls, cost=self.calls, latency=self.latency,
                    decisions=self.decisions, category=self.category, complexity=self.complexity,
                    total_reward=self.total_reward, events=events, step_calls=calls)
        return self._state(), float(reward), self.done, False, info

    def render(self):
        return f'{CATEGORIES[self.category]}: data={self.data}, analysis={self.analysis}, decisions={self.decisions}/3'

def rule_policy(state):
    _, need_data, need_analysis, data, analysis, _, _ = state
    if not need_data and not need_analysis:
        return 0
    if need_data and not data:
        return 3 if need_analysis and not analysis else 1
    return 2 if need_analysis and not analysis else 0

def workload(seed, count):
    """Balanced categories and matched independent draws per decision/operation."""
    rng = np.random.default_rng(seed)
    categories = np.arange(count) % 4
    rng.shuffle(categories)
    return dict(category=categories, complexity=rng.integers(3, size=count), draws=rng.random((count, 3, 3)))

def evaluate(policy, tasks, reliability_shift=0.0, traces=False):
    env = RoutingEnv(reliability_shift=reliability_shift)
    rows = []
    for i in range(len(tasks['category'])):
        state, _ = env.reset(seed=0, options={k: v[i] for k, v in tasks.items()})
        trace = []
        while True:
            action = int(policy(state))
            before = state.tolist()
            state, reward, done, _, info = env.step(action)
            trace.append(dict(state=before, action=action, reward=reward, events=info['events'], next_state=state.tolist()))
            if done:
                break
        row = {k: info[k] for k in ['success', 'calls', 'cost', 'latency', 'decisions', 'category', 'complexity', 'total_reward']}
        row['task_id'] = i
        if traces:
            row['trace'] = trace
        rows.append(row)
    return rows
