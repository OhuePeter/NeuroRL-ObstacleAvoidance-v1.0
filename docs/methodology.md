# Methodology

The methodology follows a train-then-perturb protocol inspired by motor-control perturbation experiments.

## Stage 1: Environment definition

- Two-dimensional reaching workspace.
- Fixed start and goal locations.
- Central obstacle geometry that enforces non-trivial trajectories.

## Stage 2: Policy training

- PPO controller is trained for obstacle-avoiding goal reaching.
- Perturbations are disabled during training.
- Checkpoints and logs are stored under experiment output folders.

## Stage 3: Frozen-policy evaluation under perturbation

- Trained policy parameters are frozen.
- Evaluation conditions: `P0`, `L1`, `L2`, `L3`, `R1`, `R2`, `R3`.
- Perturbation is triggered once when entering the trigger region.
- Behavioural trajectories, kinematics, rewards, and latent neural states are recorded.

## Stage 4: Behavioural and neural analysis

Behavioural analysis includes:

- Success and collision outcomes
- Episode duration and path-length effects
- Lateral error and adaptation metrics

Neural analysis includes:

- PCA geometry of latent states
- Condition-level representational similarity
- Perturbation-condition decoding performance
- Success-versus-failure latent separation