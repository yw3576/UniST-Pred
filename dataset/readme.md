## Data
You can download the data at [Google drive](https://drive.google.com/drive/folders/1IryA0_cDQiHfqVa9g55DJfVigExTQRuZ?usp=drive_link) and put the data in the `dataset/` folder
### SimSF-Bay
```
|----SimSF-Bay\
|    |----train_s_simsf.npy    # training and validation data for spatial block
|    |----train_t_simsf.npy    # training and validation data for temporal block
|    |----train_y_simsf.npy    # training and validation targets
|    |----test_s_simsf.npy     # testing data for spatial block
|    |----test_f_simsf.npy     # testing data for temporal block
|    |----test_y_simsf.npy     # testing targets
|    |----A_train.npy          # candidate graph structures
```

### PEMS-Bay
```
|----PEMS-Bay\
|    |----PEMS-bay.npy        # all data
|    |----adj_mx_bay.pkl      # predefined graph structure
|    |----pems-bay-meta.h5    # meta file
|    |----pems_lanes.npy      # num of lanes
```

### NYCTaxi
```
|----NYCTaxi\
|    |----train.npz    # training data
|    |----adj_mx.npz   # predefined graph structure
|    |----test.npz     # test data
|    |----val.npz      # validation data
```
