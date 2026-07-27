# Changelog

This file records additions and changes to the repository's editorial taxonomy. Dates refer to repository updates, not necessarily paper publication dates.

## Unreleased

### Added

- A concise `Start Here` introduction to the credit-assignment problem and repository scope.
- A constraint-driven method decision guide for reasoning, coding, web/GUI, and multi-agent settings.
- A 74-entry JSON catalog, generated CSV export, and catalog consistency validator.
- PivoARL as a core Agentic RL method for pivotal-aware cross-episode credit assignment.
- Standalone contribution guidelines, a paper-submission issue form, and a pull request template.
- Prepared release notes for the first dated repository release.

### Changed

- Reframed the README opening around the research question and concrete repository utilities.
- Replaced the stale `47+` summary with separate survey-snapshot, recent-entry, and benchmark counts.

## 2026.07 - 2026-07-22

### Added

- One backfilled core method: GRPO-lambda.
- Eight core methods: DelTA, SCRL, TRIAGE, OAR, CRAFT, SC-GRPO, GRAIL, and VPR.
- Three near-core methods: APPO, OPID, and PAPO.
- Progress Advantage as a related signal and QVal as an evaluation resource.
- Three related research threads: STARE, Temporal Scheduling for RLVR, and SDAR.

### Editorial Decisions

- Defined core methods as methods that directly change how sparse outcome signal is assigned to tokens, turns, actions, or agents.
- Kept Progress Advantage outside the core taxonomy because it provides an attribution signal rather than policy credit allocation.
- Kept QVal outside the method taxonomy because it evaluates dense supervision signals.
- Classified VPR as core verifiable-feedback shaping while distinguishing it from general token reweighting.

## 2026.05

### Added

- Nine entries covering turn-based policy optimization, coding-agent execution feedback, entropy and uncertainty control, orchestration traces, structured action credit, rubric rewards, and related game-theoretic attribution.

## 2026.04

### Added

- Initial repository and survey taxonomy with 47 credit-assignment methods.
- Reasoning RL, agentic RL, multi-agent, benchmark, and background sections.
