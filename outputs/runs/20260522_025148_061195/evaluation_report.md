# Federated Learning Strategies Evaluation Report
## Overview
- **Run ID:** 20260522_025148_061195
- **Total Clients:** 70
- **Rounds:** 60
- **Epochs per round:** 3
- **Evaluation split:** held-out test split
- **Sequence policy:** item_id-grouped windows; no item or split boundary crossing
- **Scaler policy:** X and y scalers fit on train rows only
- **Feature selection:** ANOVA fit on train rows only

## Results
- **Local**: RMSE = 25.7349, SMAPE = 51.0337, MAE = 17.2797, WMAPE = 0.3813, MASE = 3.1938
- **Global FedAvg**: RMSE = 24.9447, SMAPE = 51.6872, MAE = 17.3783, WMAPE = 0.3706, MASE = 3.1290
- **PA-CFL**: RMSE = 23.5801, SMAPE = 49.4666, MAE = 15.9233, WMAPE = 0.3564, MASE = 2.9765
