| model | aggregator | attack | optimizer | final acc (mean ± std) | seeds | drop vs none |
|---|---|---|---|---|---|---|
| resnet18 | mean | lie z=0.25 | Momentum SGD | 61.4 ± 0.0 | 1 | -0.7 pp |
| resnet18 | mean | lie z=0.5 | Momentum SGD | 61.9 ± 0.0 | 1 | -1.2 pp |
| resnet18 | mean | lie z=1.0 | Momentum SGD | 60.9 ± 0.0 | 1 | -0.2 pp |
| resnet18 | mean | lie z=2.0 | Momentum SGD | 55.9 ± 0.0 | 1 | +4.8 pp |
| resnet18 | mean | lie z=3.0 | Momentum SGD | 50.4 ± 0.0 | 1 | +10.3 pp |
| resnet18 | mean | minmax | Momentum SGD | 29.2 ± 0.0 | 1 | +31.4 pp |
| resnet18 | mean | minsum | Momentum SGD | 40.1 ± 0.0 | 1 | +20.6 pp |
| resnet18 | mean | none | Momentum SGD | 60.7 ± 0.0 | 1 | +0.0 pp |
| resnet18 | median | lie z=0.25 | Momentum SGD | 60.3 ± 0.0 | 1 | +1.4 pp |
| resnet18 | median | lie z=0.5 | Momentum SGD | 55.9 ± 0.0 | 1 | +5.8 pp |
| resnet18 | median | lie z=1.0 | Momentum SGD | 54.3 ± 0.0 | 1 | +7.4 pp |
| resnet18 | median | lie z=2.0 | Momentum SGD | 54.3 ± 0.0 | 1 | +7.4 pp |
| resnet18 | median | lie z=3.0 | Momentum SGD | 54.5 ± 0.0 | 1 | +7.2 pp |
| resnet18 | median | minmax | Momentum SGD | 42.9 ± 0.0 | 1 | +18.8 pp |
| resnet18 | median | none | Momentum SGD | 61.7 ± 0.0 | 1 | +0.0 pp |
| resnet18 | trimmed_mean | lie z=0.25 | Momentum SGD | 61.3 ± 0.0 | 1 | -0.5 pp |
| resnet18 | trimmed_mean | lie z=0.5 | Momentum SGD | 61.3 ± 0.0 | 1 | -0.5 pp |
| resnet18 | trimmed_mean | lie z=1.0 | Momentum SGD | 57.7 ± 0.0 | 1 | +3.0 pp |
| resnet18 | trimmed_mean | lie z=2.0 | Momentum SGD | 55.1 ± 0.0 | 1 | +5.7 pp |
| resnet18 | trimmed_mean | lie z=3.0 | Momentum SGD | 55.0 ± 0.0 | 1 | +5.7 pp |
| resnet18 | trimmed_mean | minmax | Momentum SGD | 43.9 ± 0.0 | 1 | +16.9 pp |
| resnet18 | trimmed_mean | none | Momentum SGD | 60.7 ± 0.0 | 1 | +0.0 pp |
