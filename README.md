# wavenet-imp
Pruning WaveNet architecture neural amp models using iterative magnitude pruning

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate wavenet-imp
```
## Usage

### Training (standard):

```bash
python train.py --model_cfg cfg/model/ch16_ungated.json
```

**Notes:**
- Checkpoints and logs are saved to `checkpoints/<run_stamp>/`.
- The best model (based on validation loss) is saved in the run folder.
- Each run folder will also contain `source.wav`, `target.wav`, `model_output.wav`, the model checkpoint (`.pt` file), log files, and per-epoch loss history.
- There are many configuration options you can adjust—see the argument parser in `train.py` for details. Default arguments match those used in the paper.

### Training with pruning (IMP):

```bash
python train_imp.py --model_cfg cfg/model/ch16_ungated.json
```

**Notes:**
- Checkpoints and logs are saved to `checkpoints/<run_stamp>/`.
- The best model (based on validation loss) is saved in the run folder.
- Each run folder will also contain `source.wav`, `target.wav`, `model_output.wav`, the model checkpoint (`.pt` file), log files, and per-epoch loss history.
- There are many configuration options you can adjust—see the argument parser in `train.py` for details. Default arguments match those used in the paper.

### Evaluation:

```bash
python eval.py --model_path models/ch16_ungated-best.pt --input_wav data/example/input.wav
```
Notes:
- Output is written to `outputs/<input filename>.wav` by default

### Print Contents:

```bash
python models/model_info.py models/ch16_ungated-best.pt
```

## `models/` directory

Contains trained models and their audio examples. Each run folder holds:

- `source.wav` — NAM input fed into the model
- `target.wav` — amp output captured by microphone
- `model_output.wav` — WaveNet prediction
- `losses.json` — per-epoch train/val loss history
- `logs.txt` — stdout/print logs from the run
- `*.pt` — PyTorch weights of the best-performing epoch

### Subdirectories

- `amp_captures/` — 24 amp-device captures, each trained at 90% sparsity.
- `one_shot_sweep/` — one-shot pruning applied to a single 0% model at sparsity levels from 5% to 100% (in 5% steps). Same base 0% model as `sparsity_level_sweep`.
- `prune_end_sweep/` — sweep over the pruning end epoch (`pe`).
- `prune_type_schedule/` — sweep over prune type (global/local) and prune schedule (linear/exponential).
- `sparsity_level_sweep/` — sparsity-level sweep using the best configuration found above.

### Run naming scheme

```
[model_name-]b{batch}-lr{lr}-e{epochs}-p{sparsity%}-{prune_type}-{schedule}-ps{prune_start}-pe{prune_end}-{timestamp}
```

- `model_name` — amp name prefix (only present in `amp_captures/`); omitted elsewhere (`output-...`)
- `b` — batch size
- `lr` — learning rate
- `e` — total epochs
- `p` — target sparsity (%)
- `prune_type` — `global` or `local`
- `schedule` — `linear` or `exponential`
- `ps` — prune-start epoch
- `pe` — prune-end epoch
- `timestamp` — `YYYY-MM-DD_HH-MM-SS-ffffff`

Unpruned runs omit the `p{...}-...-pe{...}` block (e.g. `output-b40-lr0.001-e1500-{timestamp}`).