# Contributing

Contributions are welcome when they improve the accuracy, scope, or usability of the curated list. Every submitted entry must be personally checked by the contributor; unreviewed bulk or fully generated additions will not be accepted.

## What Belongs Here

A paper is **core** when it directly changes how an outcome signal is assigned to tokens, segments, steps, turns, actions, or agents in LLM reinforcement learning.

Near-core optimization, related attribution signals, and evaluation resources may be included when their relationship to credit assignment is explicit. General RLHF, reward-modeling, prompting, or agent papers without a clear credit-assignment contribution are out of scope.

## Suggest A Paper

Use the paper-submission issue form and provide:

- Official title and paper URL.
- Publication year and venue or submission status, if known.
- Reasoning RL, agentic RL, multi-agent RL, or related setting.
- Proposed scope, granularity, and methodology.
- A short explanation of what signal is assigned to which decision unit.
- Official code URL, if verified.
- Any uncertainty or conflict of interest.

Being an author of the paper is not a conflict, but please disclose it.

## Submit A Pull Request

1. Add or revise the curated entry in `README.md`.
2. Add or revise the corresponding record in `data/papers.json`.
3. Run `python3 scripts/build_catalog.py` to regenerate `data/papers.csv`.
4. Run `python3 scripts/build_catalog.py --check`.
5. Explain the classification decision in the pull request.

One paper per pull request is preferred. A batch is appropriate when it represents one systematic update with consistent review criteria.

## Editorial Standards

- Use the paper's official title and canonical arXiv, OpenReview, or proceedings URL.
- Describe the mechanism and relevance, not marketing claims.
- Keep Core, Near-Core, Related Signal, and Evaluation labels distinct.
- Do not infer code availability from an unofficial reproduction.
- Do not assign a subjective quality or evidence score without a published rubric.
- End entry descriptions with a period and preserve the surrounding format.
- Keep unrelated formatting or taxonomy changes out of a paper-only pull request.

Classification is editorial and may change as papers are revised or stronger evidence becomes available.
