# Speech Commands Unknown-Handling Experiments

This repository contains a PyTorch research codebase for the Speech Commands v0.01 keyword-spotting study described in [docs/project_plan.tex](/Users/michaliwaniuk/Desktop/dl-2/docs/project_plan.tex). It treats `data/` as the canonical local dataset root and supports:

- 12-class evaluation: `yes, no, up, down, left, right, on, off, stop, go, unknown, silence`
- three backbones: `BC-ResNet-3`, `KWT-1`, `MatchboxNet-3x1x64`
- four unknown-handling strategies: `A`, `B`, `C`, `D`
- Stage 1 backbone/strategy comparison
- Stage 2 augmentation grid for the winning strategy per backbone

## Data layout

The raw Speech Commands v0.01 dataset is already included in this repository:

```text
data/
  README.md
  LICENSE
  validation_list.txt
  testing_list.txt
  audio/
    _background_noise_/
    yes/
    no/
    ...
  prepared/
```

`data/audio/` is never rewritten. Generated manifests and cached metadata are stored under `data/prepared/`.

## Installation

Create an environment with a torch-supported Python version (`3.9`-`3.12` recommended) and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For GPU training on an NVIDIA T4, install the CUDA-compatible PyTorch and torchaudio wheels appropriate for your environment before or during the editable install.

Local validation in this repository showed `torch` aborting under Python `3.13`, so avoid `3.13` for reproducible runs and test execution.

## Repository structure

```text
configs/               Experiment YAMLs and manifests
data/                  Repo-local dataset and prepared metadata
docs/                  Project documentation
outputs/               Run artifacts and aggregated summaries
scripts/               Thin entry-point wrappers
src/speech_kws/        Implementation package
tests/                 Unit and smoke tests
```

Key modules:

- `speech_kws.data.prepare`: dataset indexing, split parsing, silence metadata generation
- `speech_kws.data.dataset`: waveform loading and strategy-aware labels
- `speech_kws.audio.frontend`: log-mel and MFCC frontends
- `speech_kws.models`: backbones and wrappers
- `speech_kws.training`: loops, losses, optimizers, schedulers
- `speech_kws.evaluation`: metrics, threshold sweep, aggregation

## Configuration format

Each runnable experiment is defined by a YAML file. Common top-level sections:

- `experiment`: id, stage, seed, device, AMP
- `paths`: `audio_root`, split files, prepared metadata root, output root
- `task`: target keywords, strategy, silence policy
- `frontend`: feature extraction choice and parameters
- `augment`: time shift, background mixing, SpecAugment
- `model`: backbone name and architecture settings
- `train`: stopping mode, optimizer, scheduler, batch size, accumulation
- `strategy`: method-specific parameters such as `lambda_head2` or `tau_sweep`
- `evaluation`: metric settings and confusion-matrix export
- `outputs`: overwrite/skip policy

`base_configs` is supported for shallow inheritance. Stage 1 configs compose the common defaults with a backbone-specific file and then override the strategy-specific fields explicitly.

## CLI and scripts

Prepare dataset metadata from the existing `data/` folder:

```bash
python scripts/prepare_dataset.py
```

Run one experiment YAML:

```bash
python scripts/run_manifest.py --config configs/experiments/stage1/bcresnet_strategy_b.yaml
```

Run the full Stage 1 manifest:

```bash
python scripts/run_manifest.py --manifest configs/manifests/stage1.txt
```

Generate the Stage 2 manifest from Stage 1 results:

```bash
python scripts/build_stage2_manifest.py
```

Run the Stage 2 manifest:

```bash
python scripts/run_manifest.py --manifest configs/manifests/stage2.txt
```

Aggregate final results:

```bash
python scripts/aggregate_results.py
```

You can also use the installed CLI:

```bash
speech-kws prepare
speech-kws run --config configs/experiments/stage1/bcresnet_strategy_b.yaml
speech-kws run-manifest --manifest configs/manifests/stage1.txt
speech-kws build-stage2
speech-kws aggregate
```

## End-to-end experiments

Follow this sequence to reproduce the full two-stage study required by the project plan.

1. Create the environment and install dependencies.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Prepare dataset metadata from the repo-local `data/audio/` tree.

   ```bash
   python scripts/prepare_dataset.py
   ```

   Checkpoint: `data/prepared/dataset_summary.json` should exist, together with `spoken_manifest.csv`, `silence_manifest.csv`, and `background_manifest.csv`.

3. Run all Stage 1 experiments.

   ```bash
   python scripts/run_manifest.py --manifest configs/manifests/stage1.txt
   ```

   This executes the 12 Stage 1 configs: 3 backbones x 4 unknown-handling strategies.

4. Aggregate Stage 1 results and select per-backbone winners.

   ```bash
   python scripts/aggregate_results.py
   ```

   Checkpoint:
   - `outputs/summaries/stage1/summary.csv`
   - `outputs/summaries/stage1/winners.csv`

   `winners.csv` is the input to Stage 2 generation.

5. Generate the Stage 2 augmentation study from the Stage 1 winners.

   ```bash
   python scripts/build_stage2_manifest.py
   ```

   Checkpoint:
   - `configs/experiments/stage2/` contains 24 generated configs
   - `configs/manifests/stage2.txt` contains the generated manifest entries

   `configs/experiments/stage2/` and `configs/manifests/stage2.txt` start empty by design and are populated only by `build-stage2`.

6. Run the Stage 2 manifest.

   ```bash
   python scripts/run_manifest.py --manifest configs/manifests/stage2.txt
   ```

7. Aggregate final results again after Stage 2 completes.

   ```bash
   python scripts/aggregate_results.py
   ```

   Final checkpoints:
   - `outputs/summaries/stage1/summary.csv`
   - `outputs/summaries/stage1/winners.csv`
   - `outputs/summaries/stage2/summary.csv`
   - `outputs/summaries/all_runs.csv`

8. Inspect per-run artifacts under `outputs/runs/`.

   For every completed run, verify:
   - `val_metrics.json` and `test_metrics.json`
   - `per_class_metrics.csv`
   - `plots/confusion_raw.png`
   - `plots/confusion_row_normalized.png`
   - `threshold_sweep.csv` for Strategy `D`

## Reruns and overwrite behavior

By default, completed runs are skipped because `configs/_shared/common.yaml` sets `outputs.on_exists: skip`.

- Keep `skip` when resuming long manifests without recomputing finished runs.
- Change `outputs.on_exists` to `error` if you want duplicate outputs to fail fast.
- Remove or relocate an existing run directory if you want to regenerate that exact run from scratch.

## Outputs

Per-run artifacts are stored under:

```text
outputs/runs/{stage}/{experiment_id}/seed_{seed}/
```

Each completed run writes:

- `resolved_config.yaml`
- `run_info.json`
- `train_history.csv`
- `best.ckpt`
- `last.ckpt`
- `val_metrics.json`
- `test_metrics.json`
- `per_class_metrics.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `plots/learning_curve.png`
- `plots/confusion_raw.png`
- `plots/confusion_row_normalized.png`
- `threshold_sweep.csv` for Strategy `D`

Aggregated summaries are stored under:

```text
outputs/summaries/stage1/
outputs/summaries/stage2/
```

## Reproducibility notes

- seeds are explicit in every runnable config
- validation and test silence crops are deterministic and written to `data/prepared/`
- resolved configs are snapshotted per run
- Stage 1 and Stage 2 are driven by explicit manifest files

## Status

The codebase implements the end-to-end experiment system and its scaffolding. Training and tests require the dependencies listed in [pyproject.toml](/Users/michaliwaniuk/Desktop/dl-2/pyproject.toml); they are not bundled in the repository.
