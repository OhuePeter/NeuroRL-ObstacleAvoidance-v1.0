# Experimental Protocol

## Training

The PPO controller is trained to

- Reach the goal
- Avoid two static obstacles
- Select an optimal trajectory

No perturbations are applied during training.

---

## Evaluation

After training, the policy is frozen.

For every evaluation episode, one perturbation condition is randomly selected from

- P0
- L1
- L2
- L3
- R1
- R2
- R3

The perturbation is applied once when the agent enters the perturbation region.

The episode then continues until

- Goal reached
- Collision
- Timeout

The perturbation is therefore unexpected and never learned during training.

---

## Scientific Motivation

This protocol is inspired by biological motor-control experiments in which unexpected perturbations are introduced after a movement has been initiated.

Rather than studying how an agent learns perturbations, the framework investigates how a trained controller adapts online, enabling direct analysis of behavioural adaptation and neural population dynamics.

# Central Hypothesis

Unexpected perturbations will transiently alter both the behavioural trajectory and the internal neural population state of the trained controller.

Although the policy has never experienced perturbations during training, successful episodes are expected to exhibit rapid behavioural recovery accompanied by reorganization of the neural population dynamics.

Comparisons with the unperturbed reference condition will identify how adaptive behaviour emerges from changes in the latent neural representations.

---

---

## Comparison Strategy

The unperturbed condition (P0) serves as the reference trajectory for all behavioural and neural analyses.

Each perturbation condition (L1–L3 and R1–R3) will be quantitatively compared with the reference condition using:

### Behavioural Analysis

- Trajectory deviation
- Velocity profile
- Recovery time
- Time to goal
- Success rate
- Collision rate
- Route selection
- Path length

### Neural Population Analysis

- Principal Component Analysis (PCA)
- Neural population trajectories
- Population norms
- Correlation structure
- Functional clustering of latent units
- Neural state distances
- Decoder performance

This comparison framework allows behavioural adaptation to be directly related to changes in the internal neural population dynamics of the trained reinforcement learning controller following unexpected perturbations.