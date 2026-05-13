# Federated Learning Strategies Evaluation Report
## Overview
- **Run ID:** 20260513_055757_725833
- **Total Clients:** 70
- **Rounds:** 100
- **Epochs per round:** 3
- **Evaluation split:** held-out test split
- **Sequence policy:** item_id-grouped windows; no item or split boundary crossing
- **Scaler policy:** X and y scalers fit on train rows only
- **Feature selection:** ANOVA fit on train rows only

## Results
- **Local**: RMSE = 34.3990, SMAPE = 59.7860
- **Global FedAvg**: RMSE = 24.8674, SMAPE = 51.6973
- **PA-CFL**: RMSE = 24.7197, SMAPE = 50.0343
