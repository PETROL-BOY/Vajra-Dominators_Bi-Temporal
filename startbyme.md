# Vajra Dominators — Project Guide

> This document is the project handoff: it explains the goal, how to orient yourself in the repository, and how to prepare a safe GitHub upload. Details that depend on the source tree are marked for verification instead of guessed.

## Project at a glance

The repository is named **Vajra-Dominators**. The currently highlighted training entry point is `kaggle_prep/statquery_changeformer_train.py`, suggesting that the project includes a StatQuery/ChangeFormer training workflow and Kaggle-oriented preparation. Confirm the exact task, dataset, model lineage, and results from the script and experiment notes before treating this description as a formal project claim.

## End-to-end workflow

Use this sequence when taking the project from a fresh checkout to a training run:

1. **Understand the task and data.** Identify the prediction target, input modalities, label format, spatial/temporal resolution, and train/validation/test split. Record the dataset's source, license, and any access restrictions. Do not commit private or licensed data without permission.
2. **Set up the runtime.** Check the repository for dependency files and environment notes. The training script may target Kaggle, so check for Kaggle-specific paths, accelerator assumptions, and secrets before running elsewhere.
3. **Prepare the data.** Follow the preprocessing and path conventions in the project code. Confirm that image pairs and labels align and that the expected channel count and dimensions match the model.
4. **Configure training.** Read `kaggle_prep/statquery_changeformer_train.py` from top to bottom. Record its configuration, optimizer, loss, batch size, epoch count, checkpoint behavior, and output paths here or in the experiment record.
5. **Train and validate.** Run the documented entry point in the intended environment. Keep logs and checkpoints outside version control unless there is a deliberate, documented reason to publish a small artifact.
6. **Evaluate and share.** Record the evaluation split, metrics, checkpoint identity, and any caveats. A reproducible result should state the data version and configuration used.

## Repository map

The only path confirmed from the current project context is:

| Path | Purpose |
| --- | --- |
| `kaggle_prep/statquery_changeformer_train.py` | Highlighted training script; inspect it for the actual data pipeline, model construction, training loop, and runtime assumptions. |

Add the rest of the repository's important paths after reviewing the tree. In particular, document where data preparation, model definitions, evaluation, configuration, and experiment outputs live. Avoid listing generated caches and downloaded datasets as source files.

## Running the training workflow

The exact command and dependencies need to be confirmed from the repository before publishing instructions. Once verified, replace this section with the tested setup, for example:

```bash
# Create/activate the environment using the repository's dependency instructions.
# Obtain and prepare the dataset using its documented process.
# Run the training entry point with the required configuration.
python kaggle_prep/statquery_changeformer_train.py
```

Document required environment variables, dataset mount points, accelerator requirements, and expected outputs. Never put API keys, Kaggle credentials, access tokens, or private paths containing personal information in this file.

## Reproducibility record

For each reported run, capture:

- Dataset name/version and split definition
- Source revision or commit hash
- Python and key framework versions
- Model/configuration and random seed
- Training command and hardware/accelerator
- Checkpoint location or published artifact identifier
- Validation/test metrics and evaluation procedure

## GitHub upload checklist

Before the first push:

- Review `git status` and inspect every file that will be committed.
- Add a `.gitignore` for virtual environments, caches, checkpoints, model weights, datasets, logs, and local secrets as appropriate.
- Check for credentials and personal data. Remove secrets from files and rotate any credential that was already committed.
- Check large files and GitHub's current size limits; use Git LFS or an external artifact host for appropriate model artifacts.
- Confirm dataset and pretrained-weight licenses allow redistribution. Link to download instructions when files cannot be committed.
- Add a `README.md` with a concise project summary, setup instructions, data access, training/evaluation commands, results, and license.
- Choose and add a project license only if you have authority to do so; do not imply that third-party data or weights are covered by it.
- Create the GitHub repository, then add its URL as `origin`, commit the reviewed source, and push the chosen branch.

Example commands after creating an empty remote repository:

```bash
git add <reviewed-files>
git commit -m "Prepare project for GitHub"
git branch -M main
git remote add origin <github-repository-url>
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` before changing it. Do not force-push over existing remote history unless that is intentional.

## Project facts to fill in

Complete these from the source and experiment records so this guide is a reliable handoff:

- **Problem being solved:**
- **Dataset and license:**
- **Input and target format:**
- **Model architecture and code location:**
- **Environment/dependencies:**
- **Exact training command:**
- **Evaluation command and metrics:**
- **Known limitations:**
- **Maintainer/contact:**
