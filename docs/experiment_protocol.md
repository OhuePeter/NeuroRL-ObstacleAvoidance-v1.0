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

This protocol mimics biological motor control experiments in which unexpected perturbations are introduced after a motor plan has been formed.

The objective is to study behavioural adaptation and neural population dynamics rather than perturbation learning.