# Presentation narration and recording guide

Target runtime: approximately 14:00, including actual command-line execution and explanation of the demo output. Slides 1–15 are timed. Slide 16 is a reference appendix. Rehearse the script and use the time cues to allow brief pauses around figures.

The PowerPoint contains the same narration in speaker notes. On slide 13, show your recording of main.py executing at its natural speed, then explain the printed decisions. The code has no artificial delay.

## Slide 1: Reinforcement learning for task routing

0:00–0:30 (30 seconds)

This project asks whether reinforcement learning can learn to route tasks efficiently in a simulated system with specialized agents. The router chooses a direct response, SQL retrieval, analysis, or a combined route. I built a sequential environment, trained a Deep Q-Network, and compared it with rules and random routing. The presentation explains the design, measured results, and a command-line demonstration of the learned policy. All operations are synthetic, so the findings describe this controlled experiment.

## Slide 2: 1. Project Introduction

0:30–1:20 (50 seconds)

I chose this project because, in my current job, we use model routing for an AI report project. That work motivated me to explore whether reinforcement learning could learn efficient routing decisions. This study uses a separate simulation with specialized capabilities, so I can examine routing without company data or production access. The research question is whether DQN can complete tasks reliably while avoiding unnecessary calls, cost, and latency. The objectives from Part 1 were to build a reproducible simulation, train a learned routing policy, and compare it with simple baselines. The completed project follows those objectives.

## Slide 3: 2. Background and Project Context

1:20–2:05 (45 seconds)

The literature informed the project in two ways. Mnih and colleagues established DQN with experience replay and a separate target network, which provide the learning structure used here. FrugalGPT studies model cascades that balance response quality with inference cost. RouteLLM learns routing between stronger and weaker models from preference data. These routing studies motivate the quality and cost tradeoff, but their methods and environments differ from this experiment. My contribution is an educational comparison of sequential DQN routing with rules under controlled stochastic outcomes. I do not transfer published savings to this system or claim that DQN is necessary for routing.

## Slide 4: 3. Methodology

2:05–3:05 (60 seconds)

The environment generates four equally likely categories: direct response, retrieval, analysis, and retrieval followed by analysis. Complexity has three levels. Each applicable operation succeeds with probability point nine five, point eight five, or point seven five for low, medium, or high complexity. These probabilities are assumptions rather than estimates of real agents. The environment retains successful data and analysis outputs between decisions. The combined action executes SQL and then attempts analysis. Analysis requires data when the request requires retrieval. The environment charges an attempted analysis call even if missing data blocks it. Completion ends the episode, and exhausting the three-decision budget ends it as a failure.

## Slide 5: State and action design

3:05–4:00 (55 seconds)

The network receives seven normalized state values. They describe complexity, whether data and analysis are required, whether data is available, whether analysis is complete, the previous outcome, and the fraction of decisions used. The previous outcome distinguishes no earlier action, failure, and success. Intermediate completion flags tell the router what remains to do. The decision count tells it how much horizon remains. The four output values correspond to direct response, SQL, analysis, and SQL plus analysis. One combined decision counts as two specialized calls. This distinction matters because decisions limit the episode, while calls determine resource cost. Retries make this a sequential RL task rather than a one-step contextual bandit.

## Slide 6: Reward design

4:00–5:00 (60 seconds)

The reward combines completion, terminal failure, specialized calls, and simulated latency. A completed task receives ten points. A task that exhausts its budget without completion loses ten. Every specialized call costs one point, and each latency unit costs one tenth. For example, successful SQL retrieval earns eight point eight. A successful combined route earns seven point six because it makes two calls. On a retrieval-only request, that additional analysis call lowers the reward even though the task succeeds. Intermediate failures incur their processing costs but receive the failure penalty only at budget exhaustion. All evaluation policies share these weights. Latency units and call cost are synthetic, and cost equals the number of specialized calls.

## Slide 7: 4. Implementation

5:00–5:55 (55 seconds)

The implementation uses Python, Gymnasium, NumPy, and PyTorch on the CPU. The Q-network has seven inputs, two hidden layers with sixty-four units and ReLU activation, and four action values. Uniform replay stores up to ten thousand transitions. Training begins after five hundred transitions, uses batches of sixty-four, Adam optimization, and Huber loss. The target network copies the learned parameters every two hundred fifty updates. Epsilon decreases from one to point zero five over eight thousand episodes. The discount factor is point nine five. Compared with the original plan, I made intermediate state flags explicit and fixed the reward weights and horizon using the revised design. No live SQL or language-model calls occur.

## Slide 8: Experimental design

5:55–6:45 (50 seconds)

I first checked deterministic transitions and confirmed that DQN could learn all four categories. Then I compared learning rates of zero point zero zero one and zero point zero zero zero three with a shared pilot seed. Validation return selected the first setting. Each of five independent final seeds trained for ten thousand episodes. Every five hundred episodes, greedy evaluation used four hundred fixed validation tasks, and validation return selected the checkpoint. Only after all policies froze did I evaluate one thousand balanced held-out tasks. Rules retry the still-required capabilities. Random routing chooses uniformly. Policies receive matched task descriptions and outcome draws indexed by task, decision, and operation. A second evaluation reduces each operation probability by ten percentage points without retraining.

## Slide 9: 5. Results and Analysis

6:45–7:50 (65 seconds)

The chart shows greedy validation return every five hundred episodes. It does not show exploratory training rewards. The first evaluation already captures a useful policy, so these curves mainly show refinement and stabilization. Validation return remains close to eight point three, with some early variation across seeds. Four runs reached ninety-five percent of their best validation return by the first five-hundred-episode evaluation, while one reached it by one thousand episodes. That measurement is limited by the evaluation interval. Selected checkpoints occur between two thousand and three thousand five hundred episodes, although every run completes ten thousand training episodes. Full training takes about ten seconds per run on this machine, reflecting the very small simulation.

## Slide 10: Held-out policy comparison

7:50–8:55 (65 seconds)

Across five trained policies, DQN achieves ninety-nine point one four percent success, while rules achieve ninety-nine point three zero percent and random routing achieves sixty-six point five eight percent. The DQN gain over random is thirty-two point five six percentage points. The gap from rules is only point one six points. DQN uses about one point two two specialized calls per task, almost the same as rules and substantially fewer than random. Mean return also remains slightly below rules. The small cost advantage over rules does not establish that DQN is superior because success also falls slightly. Standard deviations describe variation across policy seeds on the same test workload. They are not confidence intervals across independent workloads.

## Slide 11: Performance by task category

8:55–9:45 (50 seconds)

Category results show where the remaining difficulty lies. Direct, retrieval-only, and analysis-only tasks reach success rates between ninety-nine point six and one hundred percent for DQN. The combined category reaches ninety-seven point three six percent, compared with ninety-eight percent for rules. Combined tasks require two successful components, and ordering or retries can consume the decision budget. Random routing performs especially poorly on direct and combined tasks. Aggregate success would hide this pattern. The category comparison supports the interpretation that DQN learned the basic routing requirements, while the largest residual weakness concerns the longer sequence. This environment provides explicit requirement flags, which makes a strong rule-based solution straightforward.

## Slide 12: Sensitivity to less reliable agents

9:45–10:35 (50 seconds)

In the secondary evaluation, each success probability drops by point one: the three levels become point eight five, point seven five, and point six five. The learned policies remain frozen. DQN overall success falls from ninety-nine point one four to ninety-six point nine six percent, and calls increase from one point two two to one point three nine per task. Combined-task success falls more sharply, from ninety-seven point three six to ninety-one point eight four percent. Rules remain slightly ahead overall, and random routing also deteriorates. The result shows some tolerance to lower reliability through retries, but it does not demonstrate broad generalization. Both scenarios keep the same task definitions and synthetic transition model.

## Slide 13: 6. Demonstration

10:35–12:15 (100 seconds)

This demonstration executes the saved DQN directly from the main entry point. It prints the current state, predicted action values, the chosen action, and the environment outcome as the code runs. The code uses synthetic operations and reproducible outcome draws, following the project design. There are no added delays. The measured execution time appears at the end.

[Show your recording, then explain its output.] In test task seven, the router first selects SQL retrieval, which succeeds. Analysis then fails, but the retrieved data remains available. The router selects analysis again, and the task completes. The return is six point four after three specialized calls. This example demonstrates recovery without repeating retrieval. Running main.py without a task option shows six selected cases, including a failure at the decision budget. These examples illustrate behavior, while the held-out test metrics measure overall performance.

## Slide 14: 7. Ethical Considerations

12:15–13:05 (50 seconds)

The current project uses synthetic tasks and makes no external model calls or database queries, so it avoids exposing private data during experimentation. A real deployment would introduce different risks. A router optimizing cost could choose an inadequate capability for a consequential request. SQL access would require permissions and safeguards, and routing logs could contain sensitive information. Reliability monitoring should track failures by request type and complexity, rather than relying only on an average success score. A real system would also need escalation or a safe fallback when the decision budget ends. This simulation cannot establish fairness across people or safety in production. Accountability would require an identifiable system owner and auditable routing decisions.

## Slide 15: 8. Conclusion and Future Work

13:05–14:00 (55 seconds)

The project demonstrates that a standard DQN can learn effective routing and retry behavior in this small sequential simulation. It exceeds random routing by thirty-two point five six percentage points, remains close to rules, and meets the planned cost target. The main lesson is that successful RL depends on careful environment and reward design as much as on the network. Rules remain competitive because the state directly reveals what each request needs. Future work should test noisy or hidden task requirements, dynamic capability costs, and more independent test workloads. Algorithm comparisons could include tabular Q-learning or Double DQN. Any move toward live agents would require a separate evaluation of response quality, permissions, and safe fallback behavior.

## Slide 16: References

Appendix, outside the timed narration

Reference slide for consultation after the timed presentation.

AI assistance disclosure: AI assisted with implementation, experiment execution, presentation preparation, and narration drafting. The presenter should review the code, results, and final submission and adapt this disclosure to course policy.

## Recording checklist

1. Review the code and measured results, then personalize the wording.
2. Open the PowerPoint in Presenter View to use speaker notes.
3. Record python main.py --task 7 --show-q executing at its natural speed. Insert your recording on slide 13 and explain the output.
4. Record your slides, voice, and demo together using PowerPoint, Teams, or your preferred screen recorder.
5. Rehearse once, adjust pacing to 13–15 minutes, and verify that the final recording includes the demo and your narration.

The presenter must record the final presentation and command-line demo. The code runs naturally without timed pauses.
