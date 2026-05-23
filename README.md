# UniST-Pred
<One-sentence description of what this project does.>

- **Task:** < forecasting>
- **Framework:** < PyTorch>
- **Dataset:** <SimSF-Bay, PEMS-Bay, NYCTaxi>
- **License:** <MIT / Apache-2.0 / Proprietary>
  
<img src="figures/UniST_archi.png" alt="TheTable" />

## Train and Evaluate

please run the following commands to train the model on the specific configure file from `{configure_simsf, configure_pems, configure_nyctaxi}`.
```bash
>> cd UniST_Pred
>> ./run_.sh  --config_file  configure_pems   # configure_pems specifies the configure file
```

## Data
You can download the data at [Google drive](https://drive.google.com/drive/folders/1n5gpZINmXJ_02YF7jSP_IHqmXffVyien?usp=drive_link) and put the data in the `dataset/` folder
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

## Configuration

Configs are stored in `configs/` (YAML). Sections include:

- `model`: architecture
- `data`: sequence length, target length, nodes
- `train`: learning-rate schedule, epochs, batch size

## Benchmarks
ARIMA is implemented via statsmodels python package. LSTM is implemented from Pytorch Neural Networks package. 
Other benchmarks are implemented via the code provided in their project repositories: 
[TCN](https://github.com/colincsl/TemporalConvolutionalNetworks), [TSMixer](https://github.com/smrfeld/tsmixer-pytorch?tab=readme-ov-file), [STGCN](https://github.com/hazdzz/stgcn?tab=readme-ov-file), [ASTGCN](https://github.com/wanhuaiyu/ASTGCN), [ST-SSL](https://github.com/Echo-Ji/ST-SSL), [DCRNN](https://github.com/liyaguang/DCRNN), and [STEP](https://github.com/GestaltCogTeam/STEP)

## MATSim Simulator & SF Bay Transportation Network Model
MATSim (Multi-Agent Transport Simulation) is a microscopic, agent-based transportation simulator widely utilized in transportation planning and mobility research. MATSim models travel demand explicitly at the individual level, representing each traveler as an autonomous agent with daily activity plans that evolve through iterative re-planning and network loading. At each iteration, agents execute their plans on a detailed transportation network, generating time-resolved link-level traffic states such as flow, speed, and travel time, which are subsequently used to update agent plans based on experienced costs and utilities defined by user-assigned scoring functions. The simulation is considered converged once agents' plans and experienced costs stabilize across successive iterations, corresponding to a stochastic user equilibrium. Owing to its realism, flexibility, and full control over demand, supply, and network configurations, MATSim is commonly employed by academic researchers, transportation authorities, and planning agencies for scenario analysis, policy evaluation, and infrastructure planning.

Within the scope of current study, we adopt from the MATSim model calibrated for the San Francisco Bay Area. In this model, the commuters travel plan is based on the vehicular travel information from Pozdnoukhov et al. The demand model is based on anonymised cellular network infrastructure data stream and census data from the 2010-2012 California Household Travel Survey data (available at \url{https://www.nrel.gov/transportation/secure-transportation-data/tsdc-california-travel-survey}). The demand generation relies on the 1454 Traffic Analysis Zones in the area developed by the Metropolitan Transportation Commission (see \url{https://abag.ca.gov/sites/default/files/pba_2050-regional_growth_forecast_methodology.pdf}). The model considers  a sample population of 463,938 commuters and the road links in the transportation network are scaled down to 8\% of their original capacities to correctly match the population scale.

