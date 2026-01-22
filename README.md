# UniST-Pred
<One-sentence description of what this project does.>

- **Task:** < forecasting>
- **Framework:** < PyTorch>
- **Dataset:** <SimSF-Bay, PEMS-Bay, NYCTaxi>
- **License:** <MIT / Apache-2.0 / Proprietary>
  
<img src="figures/UniST_archi.png" alt="TheTable" />

## Train and Evaluate
cd UniST_Pred \
./run_.sh configure_pems

## Data
You can download the data at [Google drive](https://drive.google.com/drive/folders/1IryA0_cDQiHfqVa9g55DJfVigExTQRuZ?usp=drive_link)
### SimSF-Bay
```
|----SimSF-Bay\
|    |----PEMS-bay.npy        # all data
|    |----adj_mx_bay.pkl      # predefined graph structure
|    |----pems-bay-meta.h5    # meta file
|    |----pems_lanes.npy      # num of lanes
```

### PEMS-Bay
```
|----PMES-Bay\
|    |----train_s_simsf.npy    # training and validation data for spatial block
|    |----train_t_simsf.npy    # training and validation data for temporal block
|    |----test_s_simsf.npy     # testing data for spatial block
|    |----test_f_simsf.npy     # testing data for temporal block
|    |----A_train.npy          # candidate graph structures
```

### NYCTaxi
```
|----NYCTaxi\
|    |----train.npz    # training data
|    |----adj_mx.npz   # predefined graph structure
|    |----test.npz     # test data
|    |----val.npz      # validation data
```

## Configuration

Configs are stored in `configs/` (YAML). Sections include:

- `model`: architecture
- `data`: sequence length, target length, nodes
- `train`: learning-rate schedule, epochs, batch size
