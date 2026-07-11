# NeuroRL-ObstacleAvoidance-v1.0

## Overview

NeuroRL-ObstacleAvoidance-v1.0 is a computational neuroscience framework for studying neural population dynamics in reinforcement learning agents during goal-directed obstacle avoidance under unexpected perturbations.

The project investigates how internal neural representations emerge and reorganize when a trained reinforcement learning controller experiences unpredictable disturbances, drawing inspiration from biological motor control and visuomotor adaptation.

---

## Scientific Motivation

Biological motor systems continuously adapt to unexpected perturbations through coordinated neural population activity.

This project investigates whether similar computational principles emerge within reinforcement learning policy networks. Perturbations are **never introduced during training**. Instead, a trained policy is evaluated under unexpected perturbations to examine behavioural adaptation and internal neural population dynamics.

---

## Main Contributions

- Goal-directed obstacle avoidance using Proximal Policy Optimization
- Unexpected perturbations during evaluation only
- Neural population recording from hidden units
- Principal Component Analysis (PCA)
- Neural correlation clustering
- Latent unit analysis
- Behavioural and neural decoding
- Publication figures

---

## Repository Structure

docs/
configs/
src/
analysis/
experiments/
paper/
presentation/
figures/
scripts/
tests/

---

## Installation

```bash
conda env create -f environment.yml
conda activate neurorl
pip install -r requirements.txt


## Funding

This work was undertaken thanks in part to funding from the Connected Minds Program, supported by the Canada First Research Excellence Fund (CFREF), Grant #CFREF-2022-00010.

## Institutional Affiliation

Centre for Neuroscience Studies
Queen's University, Kingston, Ontario, Canada

This repository was developed within the CompSci Lab at Queen's University.