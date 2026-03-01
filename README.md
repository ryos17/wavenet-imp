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
python train.py --model_cfg cfg/model/ch16_ungated.json --train_cfg cfg/train/example.json
```

**Notes:**
- Checkpoints and logs are saved to `checkpoints/<run_stamp>/`.
- Each epoch writes `{model_basename}-epoch_{epoch}.pt`, `source.wav`, `target.wav`, and `model_output.wav` in `epoch_<XXX>/`.
- The best model (by validation loss) is saved as `{model_basename}-best.pt` in the run folder.

### Training with pruning (IMP):

```bash
python train_imp.py --model_cfg cfg/model/ch16_ungated.json --train_cfg cfg/train_imp/example.json
```

**Notes:**
- Checkpoints and logs are saved to `checkpoints/<run_stamp>/`.
- Each epoch writes `prune_{sparsity}_{model_basename}-epoch_{epoch}.pt`, `source.wav`, `target.wav`, and `model_output.wav` in `epoch_<XXX>/`.
- The best model (by validation loss) is saved as `prune_{sparsity}_{model_basename}-best.pt` in the run folder.

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