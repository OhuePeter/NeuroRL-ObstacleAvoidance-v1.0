# Journal Upgrade Plan (No-AI Tone, Voice-Preserving)

## Current Position

- Nature-tier broad journals: not ready yet.
- Strong specialist/PLOS-tier journals: potentially viable after rigor upgrades.

## Three Priority Gaps (and how to close them)

### 1) Multi-seed reproducibility and uncertainty intervals

What to add:
- Train/evaluate at least 5 seeds (preferably 10).
- Report condition-wise success rate mean ± 95% CI.
- Report variability for PCA variance explained (PC1/PC2) and decoding metrics.

How to write it in your voice:
- "We repeated training across independent seeds to quantify stability rather than relying on a single realization."
- "The perturbation trend remained consistent across seeds, with uncertainty intervals that preserved the same high-rightward failure pattern."

### 2) Baseline and ablation comparisons

What to add:
- Baseline RL algorithm: SAC or TD3.
- Ablation A: remove route cue.
- Ablation B: remove route shaping term.
- Ablation C: no evaluation-time noise.

How to write it in your voice:
- "To test whether the observed robustness pattern was PPO-specific or architecture-specific, we compared against baseline controllers and targeted ablations."
- "The route-cue and route-shaping ablations reduced adaptation quality, confirming their functional role in this task."

### 3) Stronger causal framing limits

What to change in language:
- Replace "demonstrates causal mechanism" with "is consistent with" or "is associated with."
- Keep claims bounded by measurements.

Safe phrasing examples:
- "These latent changes were systematically associated with behavioural failures under stronger perturbations."
- "The analyses support an interpretable representational account, but do not by themselves establish causality."

## Additional Upgrades That Improve Acceptance Probability

1. Add one compact robustness table in main text:
- condition, success rate, collision rate, mean steps, mean lateral error.

2. Add one reproducibility figure:
- seed-wise success curves with confidence bands.

3. Add one methods transparency paragraph:
- exact package versions, random seeds, checkpoint path, and script entry points.

4. Tighten discussion claims:
- Keep biological comparisons as computational analogies, not equivalence statements.

## Voice Guardrail (to avoid AI tone)

- Use short declarative sentences.
- Keep one claim per sentence.
- Avoid promotional adjectives ("novel", "groundbreaking", "unprecedented") unless supported.
- Prefer concrete numbers over abstract emphasis.

Example conversion:
- Avoid: "Our groundbreaking framework conclusively reveals neural intelligence."
- Prefer: "The controller showed a graded robustness profile, and latent structure changed systematically with perturbation level."

## What is already implemented in this repo now

- Descriptive-statistics section removed from manuscript body.
- Refined results section with claim-safe language and direct figure anchoring.
- New figures generated for:
  - neural population PCA + trajectory dynamics,
  - hidden-unit tuning,
  - goal-variable decoding + confusion matrix,
  - route-choice distribution + perturbation distribution,
  - hierarchical latent organization.
