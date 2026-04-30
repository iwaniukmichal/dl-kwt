# Speech Commands Unknown-Handling

This repository contains a PyTorch research codebase for the Speech Commands v0.01 keyword-spotting study described in [Project Plan](docs/project_plan.pdf).

## Data layout

The raw Speech Commands v0.01 dataset should be put into data folder:

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

## Installation

Create an environment with a torch-supported Python version (`3.9`-`3.12` recommended) and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

Generate the Stage 3 KWT manifest from the best Stage 2 KWT run:

```bash
python scripts/build_stage3_manifest.py
```

Run the Stage 2 manifest:

```bash
python scripts/run_manifest.py --manifest configs/manifests/stage2.txt
```

Run the Stage 3 manifest:

```bash
python scripts/run_manifest.py --manifest configs/manifests/stage3.txt
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
speech-kws build-stage3
speech-kws aggregate
```

## End-to-end experiments

This end-to-end protocol allows results to be reproduced

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

7. Generate the Stage 3 KWT hyperparameter study from the best `kwt_strategy_a_aug_111` Stage 2 run.

   ```bash
   python scripts/build_stage3_manifest.py
   ```

   Checkpoint:
   - `configs/experiments/stage3/` contains 6 generated configs
   - `configs/manifests/stage3.txt` contains the generated manifest entries

8. Run the Stage 3 manifest.

   ```bash
   python scripts/run_manifest.py --manifest configs/manifests/stage3.txt
   ```

9. Aggregate final results again after Stage 3 completes.

   ```bash
   python scripts/aggregate_results.py
   ```

   Final checkpoints:
   - `outputs/summaries/stage1/summary.csv`
   - `outputs/summaries/stage1/winners.csv`
   - `outputs/summaries/stage2/summary.csv`
   - `outputs/summaries/stage3/summary.csv`
   - `outputs/summaries/all_runs.csv`

10. Inspect per-run artifacts under `outputs/runs/`.

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
outputs/summaries/stage3/
```

## Reproducibility notes

- seeds are explicit in every runnable config
- validation and test silence crops are deterministic and written to `data/prepared/`
- resolved configs are snapshotted per run
- Stage 1, Stage 2, and Stage 3 are driven by explicit manifest files
