# Reinforcement learning for subagent routing

This project trains a Deep Q-Network (DQN) to route tasks between simulated subagents. The router can choose a direct response, a SQL subagent, an analysis subagent, or both subagents in sequence.

The SQL role represents retrieval, and the analysis role represents interpretation. Their outputs are completion flags. The project does not call live language models or databases. Saved checkpoints contain the trained router.

## Setup on Windows

The project was tested with Python 3.12.2. Open PowerShell in this folder and create a virtual environment:

```powershell
python -m venv .venv
```

Install the dependencies using the environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can run the commands below without activating the environment.

## Run the demonstration

Run all six selected examples:

```powershell
.\.venv\Scripts\python.exe main.py
```

For the three examples used in the presentation, run:

```powershell
.\.venv\Scripts\python.exe main.py --task 18 --show-q
.\.venv\Scripts\python.exe main.py --task 7 --show-q
.\.venv\Scripts\python.exe main.py --task 0 --show-q
```

- Task 18 shows successful retrieval with one subagent call.
- Task 7 shows retrieval, failed analysis, and a successful analysis retry.
- Task 0 shows failure when the three-decision budget runs out.

These numbers are IDs in the saved test workload. The router receives state values, not task IDs. The `--show-q` option prints the predicted value of each route. The router selects the route with the highest Q-value.

The program prints decisions and outcomes as the code executes. There are no artificial delays. Saved random numbers make subagent success and failure reproducible. The elapsed-time line measures checkpoint loading and demo execution, excluding Python startup and imports. Simulated latency units are a separate resource measure.

If the dependencies are already installed in your normal Python environment, you can use `python main.py` instead.

## Train and evaluate the project

The saved checkpoints are enough to run the demonstration. To reproduce the experiment from the beginning:

```powershell
.\.venv\Scripts\python.exe experiment.py
```

This runs the deterministic learning check, compares two learning rates, trains five final DQN runs, and evaluates DQN, rule-based, and random policies. It also tests reduced subagent reliability.

Training writes configurations, logs, checkpoints, workloads, and results to `results/`. It overwrites files with the same names, so preserve the supplied results before rerunning if you need the original experiment evidence. Reproduction may differ slightly with different package versions or hardware.

## Command options

| Option | Purpose |
|---|---|
| `--task ID` | Run one saved test task, from 0 to 999. |
| `--show-q` | Print the four predicted route values. |
| `--seed 11` | Choose a trained checkpoint. Available seeds: 11, 22, 33, 44, 55. |
| `--shift -0.1` | Reduce each operation's success probability by 0.10 without retraining. |
| `--presentation` | Run the six selected examples, which is also the default. |

## Files needed to run

- `main.py`: runs the trained router and prints the demonstration.
- `experiment.py`: defines the DQN, trains it, evaluates policies, and loads checkpoints.
- `routing_env.py`: defines the simulated environment, rewards, workloads, and baselines.
- `requirements.txt`: lists Python dependencies.
- `results/checkpoints/dqn_*.pt`: trained routers for the five seeds.
- `results/test_workload.npz`: saved test tasks and random outcome values.
- `results/selected_demo_episodes.json`: identifies the six demonstration tasks.
- Other files in `results/`: configurations, training logs, and measured results supporting the presentation.

## Record the demonstration

Maximize the terminal and increase the font size. Record the three commands above running at their natural speed, then explain the output. Include the screen recording in the Demonstration slide. The presenter must record the full presentation and narration separately.

## Git

The `.gitignore` excludes virtual environments, Python caches, temporary Office files, local build files, logs, and ZIP copies. Trained checkpoints, saved workloads, result CSVs, and presentation files remain eligible for tracking so the demonstration can run after downloading the project.
