# Self-supervised Vision Transformers for Land-cover Segmentation and Classification

## Data
The datasets used in this work are publicly available:
* [SEN12MS](https://mediatum.ub.tum.de/1474000)
* [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files)

## Checkpoints
We provide checkpoints for ResNet50 and Swin-t transformer models trained with Sentinel-1/2 pairs from SEN12MS (self-supervised):
* [ResNet50]()
* [Swin-t]()

## Code
This repository uses code from the following sources:
* [Data handling](https://github.com/lukasliebel/dfc2020_baseline)
* [Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL)
* [SimCLR](https://github.com/sthalles/SimCLR)
