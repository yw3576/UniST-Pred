# UniST-Pred
<One-sentence description of what this project does.>

- **Task:** <forecasting>
- **Framework:** <PyTorch>
- **Dataset:** <SimSF-Bay, PEMS-Bay, NYCTaxi>
- **License:** <MIT / Apache-2.0 / Proprietary>

## Train
python test_unistpred.py --config configs/[configure file name]
## Evaluate

## Data



## Configuration

Configs are stored in `configs/` (YAML). Sections include:

- `model`: architecture
- `data`: sequence length, target length, nodes
- `train`: learning-rate schedule, epochs, batch size
