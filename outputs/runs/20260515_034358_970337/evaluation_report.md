# Federated Learning Strategies Evaluation Report
## Overview
- **Run ID:** 20260515_034358_970337
- **Total Clients:** 70
- **Rounds:** 100
- **Epochs per round:** 3
- **Evaluation split:** held-out test split
- **Sequence policy:** item_id-grouped windows; no item or split boundary crossing
- **Scaler policy:** X and y scalers fit on train rows only
- **Feature selection:** ANOVA fit on train rows only

## Results
- **Local**: RMSE = 34.5582, SMAPE = 60.1436, MAE = 22.8122, WMAPE = 0.4958, MASE = 4.1788
- **Global FedAvg**: RMSE = 24.8726, SMAPE = 51.6922, MAE = 17.2438, WMAPE = 0.3706, MASE = 3.1214
- **PA-CFL**: RMSE = 25.5390, SMAPE = 50.7131, MAE = 16.9697, WMAPE = 0.3724, MASE = 3.1238
