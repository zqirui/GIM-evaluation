## Evaluation Framework for Generative Image Models
This repository contains the code for my master thesis on the Evaluation of Generative Image Models.
It features an image based evaluation framework with the following metrics implemented.

## Metrics and Packages
| Metric/Evaluation                                     | Reference                                                         | Python Package / Repository                                                                    |
|-------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Inception Score (IS)                                  | [Salimans et al., 2016](https://arxiv.org/pdf/1606.03498.pdf)     | [torch-fidelity](https://github.com/toshas/torch-fidelity)                                     |
| Fréchet Inception Distance (FID)                      | [Heusel et al., 2018](https://arxiv.org/pdf/1706.08500.pdf)       | [torch-fidelity](https://github.com/toshas/torch-fidelity)                                     |
| Kernel Inception Distance (KID)                       | [Binkowski et al., 2018](https://arxiv.org/pdf/1801.01401.pdf)    | [torch-fidelity](https://github.com/toshas/torch-fidelity)                                     |
| Unbiased IS $`IS_{\infty}`$                           | [Chong & Forsyth, 2020](https://arxiv.org/pdf/1911.07023.pdf)     | [FID_ID_infinity](https://github.com/mchong6/FID_IS_infinity)                                  |
| Unbiased FID $`FID_{\infty}`$                         | [Chong & Forsyth, 2020](https://arxiv.org/pdf/1911.07023.pdf)     | [FID_ID_infinity](https://github.com/mchong6/FID_IS_infinity)                                  |
| CleanFID                                              | [Parmar et al., 2022](https://arxiv.org/pdf/2104.11222.pdf)       | [clean-fid](https://github.com/GaParmar/clean-fid)                                             |
| CleanKID                                              | [Parmar et al., 2022](https://arxiv.org/pdf/2104.11222.pdf)       | [clean-kid](https://github.com/GaParmar/clean-fid)                                             |
| Precision-Recall (PRD)                                | [Sajjadi et al., 2018](https://arxiv.org/pdf/1806.00035.pdf)      | [precision-recall-distributions](https://github.com/msmsajjadi/precision-recall-distributions) |
| Alpha/Beta Precision-Recall                           | [Kynkäänniemi et al., 2019](https://arxiv.org/pdf/1904.06991.pdf) | [torch-fidelity](https://github.com/toshas/torch-fidelity)                                     |
| Likeliness Score (LS)                                 | [Guan & Lowe, 2021](https://arxiv.org/pdf/2002.12345.pdf)         | [GAN_evaluation_LS](https://github.com/ShuyueG/GAN_evaluation_LS)                              |
| Classifier Two-Sample Test (C2ST): 1 Nearest Neighbor | [Lopez-Paz & Oquab, 2018](https://arxiv.org/pdf/1610.06545.pdf)   | Self implemented                                                                               |
| Memorization-informed FID                             | [Bai et al., 2021](https://arxiv.org/pdf/2106.03062.pdf)                                              | Self implemented                                                                               |

## How to use
To specify the metrics, hyperparameter and feature extractor the `EvalConfig` and for other system variables the `PlatformConfig` is used. 
The `EvalConfig` specifies which metrics are computed and also contains the hyperparameter for each metric.
Please refer the `EvalConfig` (configs.py) for further information on the (default) parameter.

Example:
```python
from framework.configs import PlatformConfig, EvalConfig, FeatureExtractor

eval_cfg = EvalConfig(
    feature_extractor=FeatureExtractor.InceptionV3,
    c2st_knn=False,
    fid_infinity=False,
    is_infinity=False,
    clean_fid=False,
    clean_kid=False,
    inception_score=False,
    fid=False,
    kid=False,
    prc=False,
    ls=True,
    mifid=False,
    prd=False,
)

platform_cfg = PlatformConfig(
    verbose=True,
    cuda=True,
    save_cpu_ram=True,
    compare_real_to_real=True,
    num_worker=16,
)
```

`EvalConfig`,`PlatformConfig` and the path to the real images are passed to the `PlatformManager`.
Calling `calc_metrics()` will start the metric computation.
The results are returned as a custom result dictionary (`ResultDict`) which allows reading and writing to/from JSON.

```python
from framework.platform import PlatformManager

platform_manager = PlatformManager(real_images_path=...,
                                   eval_config=eval_cfg,
                                   platform_config=platform_cfg)
                                   
result_dict = platform_manager.calc_metrics()
result_dict.print()                                
```