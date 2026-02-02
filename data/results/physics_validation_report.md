# Physics Validation Report

## Overall Metrics
| Parameter | MAE | RMSE | Bias | MAPE |
|---|---|---|---|---|
| MMAD | 1.004 | 1.586 | 0.562 | 221.9% |
| Concentration | 1134.588 | 2769.619 | -313.356 | 94.4% |
| GSD | 0.421 | 0.534 | 0.060 | 21.5% |

## Dataset-Level Comparison
| geo_id    |   True_MMAD |   Pred_MMAD |   True_Concentration |   Pred_Concentration |
|:----------|------------:|------------:|---------------------:|---------------------:|
| GSE10006  |        0.45 |    0.850135 |                    0 |              494.239 |
| GSE18385  |        0.45 |    1.1824   |                    0 |              481.677 |
| GSE237251 |        2    |    0.642117 |                11000 |              863.341 |
| GSE25531  |        0.3  |    1.86069  |                  300 |               11.251 |