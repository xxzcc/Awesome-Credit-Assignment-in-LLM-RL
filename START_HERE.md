# Start Here: Credit Assignment in LLM Reinforcement Learning

Credit assignment asks a narrower question than reward design:

> After an LLM receives an outcome reward, which token, reasoning step, interaction turn, action, or agent should receive credit or blame?

A scalar terminal reward says whether a trajectory succeeded. It does not explain which decisions caused that outcome. This gap becomes harder as reasoning chains grow longer, agents interact with partially observed environments, and multiple agents share one team reward.

## The Two Axes

The repository organizes methods along two independent axes.

| Axis | Main question | Typical choices |
|---|---|---|
| Granularity | Where is credit assigned? | Token, segment, step, turn, action, agent. |
| Methodology | How is credit estimated? | Monte Carlo, temporal difference, critic, counterfactual, game-theoretic, information-theoretic, uncertainty control, verifiable feedback. |

Granularity determines the unit that receives an advantage or reward. Methodology determines the evidence used to estimate that advantage.

## Two Regimes

### Reasoning RL

A single generation may contain hundreds or thousands of tokens before a verifiable answer. The main challenge is locating useful reasoning steps without rewarding fluent but causally irrelevant text.

Start with the [reasoning RL method sections](README.md#credit-assignment-in-reasoning-rl) when your environment is mathematical reasoning, code generation, or another single-response task.

### Agentic RL

An agent acts across many turns while tools and environments change the state. The main challenge is distinguishing useful actions from recovery steps, redundant exploration, and failures caused by earlier decisions.

Start with the [agentic RL method sections](README.md#credit-assignment-in-agentic-rl) for web, GUI, coding, search, tool-use, and other interactive agents.

For shared rewards across several LLM agents, use the [multi-agent section](README.md#multi-agent-credit-assignment).

## Four Questions To Ask First

1. **What feedback exists?** Terminal outcome only, intermediate verifier, execution result, process label, or learned judge?
2. **Can the system branch?** Can it sample continuations or sibling rollouts from intermediate states?
3. **Can it train or call a critic?** A value model or LLM critic can provide dense estimates but adds cost and possible bias.
4. **Who shares the reward?** One sequence, one long-horizon agent, or several collaborating agents?

These questions narrow the method family more reliably than selecting by benchmark name alone. Continue with the [method decision guide](method_decision_tree.md).

## Repository Scope

- **Core** methods directly change how outcome signal is assigned to tokens, turns, actions, or agents.
- **Near-Core** methods alter closely related optimization signals but are not pure credit-assignment methods.
- **Related Signal** entries provide useful attribution or progress signals without directly defining policy credit.
- **Evaluation** entries test dense supervision signals or credit estimators.

Use the [machine-readable catalog](data/README.md) for filtering and analysis, and [Recent Additions](README.md#recent-additions) for the latest editorial update.
