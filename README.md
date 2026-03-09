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