# Fork Guide

This guide provides a clean and reproducible workflow for creating and maintaining a fork.

## 1. Fork on GitHub

1. Open the repository page.
2. Click Fork.
3. Create the fork under your account.

## 2. Clone your fork

```bash
git clone https://github.com/<your-user>/NeuroRL-ObstacleAvoidance-v1.0.git
cd NeuroRL-ObstacleAvoidance-v1.0
```

## 3. Add upstream remote

```bash
git remote add upstream https://github.com/OhuePeter/NeuroRL-ObstacleAvoidance-v1.0.git
git fetch upstream
```

Check remotes:

```bash
git remote -v
```

## 4. Keep your fork up to date

From your main branch:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

If your default branch is not main, replace main with your branch name.

## 5. Create a feature branch

```bash
git checkout -b feature/<short-description>
```

## 6. Commit and push

```bash
git add .
git commit -m "Describe your change"
git push -u origin feature/<short-description>
```

## 7. Open pull request

Open a pull request from your fork branch to the upstream target branch and include:

- Scientific or technical motivation.
- Files changed.
- Validation commands run.
- Expected outputs and output paths.
