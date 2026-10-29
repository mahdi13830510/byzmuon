| model | aggregator | attack | optimizer | final acc (mean ± std) | seeds | drop vs none |
|---|---|---|---|---|---|---|
| resnet18 | mean | lie | Momentum SGD | 61.4 ± 0.0 | 1 | -0.7 pp |
| resnet18 | mean | none | Momentum SGD | 60.7 ± 0.0 | 1 | +0.0 pp |
| resnet18 | median | lie | Momentum SGD | 60.3 ± 0.0 | 1 | +1.4 pp |
| resnet18 | median | none | Momentum SGD | 61.7 ± 0.0 | 1 | +0.0 pp |
| resnet18 | trimmed_mean | lie | Momentum SGD | 61.3 ± 0.0 | 1 | -0.5 pp |
| resnet18 | trimmed_mean | none | Momentum SGD | 60.7 ± 0.0 | 1 | +0.0 pp |
