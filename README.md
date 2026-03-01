# wavenet-imp
Pruning WaveNet architecture neural amp models using iterative magnitude pruning

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate wavenet-imp
```
## Usage

### Training:

```bash
python train.py --model_cfg cfg/model/ch16_ungated.json --train_cfg cfg/train/example.json
```
Notes:
- Checkpoints are saved to `checkpoints/<timestamp>/`.
- Each epoch writes `checkpoint.pt`, `source.wav`, `target.wav`, and `model_output.wav` in `epoch_XXX/`.
- Best model by validation loss is saved as `best.pt`.

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