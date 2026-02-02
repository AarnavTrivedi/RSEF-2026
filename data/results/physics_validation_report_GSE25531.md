# Physics Validation Report: GSE25531 (Gold Standard)

> **Dataset**: GSE25531 (Human Diesel Exhaust)
> **Condition**: 300 μg/m³, 0.3 μm MMAD

## Overall Metrics
| Parameter | MAE | RMSE | Bias | MAPE |
|---|---|---|---|---|
| MMAD | 0.304 | 0.363 | -0.024 | 101.3% |
| Concentration | 1482.770 | 2217.660 | 1124.379 | 494.3% |
| GSD | 0.529 | 0.579 | -0.289 | 24.0% |

## Dataset-Level Comparison
| geo_id   |   True_MMAD |   Pred_MMAD |   True_Concentration |   Pred_Concentration |
|:---------|------------:|------------:|---------------------:|---------------------:|
| GSE25531 |         0.3 |    0.276041 |                  300 |              1424.38 |

## Per-Sample Predictions
|    | geo_id   |   True_MMAD |   Pred_MMAD |   True_Concentration |   Pred_Concentration |
|---:|:---------|------------:|------------:|---------------------:|---------------------:|
|  9 | GSE25531 |         0.3 |   0.119556  |                  300 |              1.02501 |
| 11 | GSE25531 |         0.3 |   0.0963173 |                  300 |           2317.73    |
| 14 | GSE25531 |         0.3 |   0.0606778 |                  300 |           4800.14    |
| 18 | GSE25531 |         0.3 |   0.999924  |                  300 |              1.18863 |
| 51 | GSE25531 |         0.3 |   0.103728  |                  300 |              1.80879 |