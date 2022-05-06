# Self-supervised Vision Transformers for Land-cover Segmentation and Classification

## Self-supervised Training 
### ResNet50
The script `train_resnet_backbone.py` trains a model with two ResNet50 backbones for Sentinel-1/2 inputs. The the path to the SEN12MS dataset can be supplied with the `--train_dir` flag.
### Swin-t
The Swin transformer backbones are trained with the `train_d_swin_backbone.py` script. The model code is adapted from 
[here](https://github.com/SwinTransformer/Transformer-SSL). Changes to the default parameters, e.g. path to the training data, can be made by adjusting the values in the config file at `configs/backbone_config.json`.

## Classification Downstream Tasks
Fine-tuning and linear evaluation on the classification tasks is done with `train_evaluation.py`. The `downstream_eval_scripts/` directory contains small bash scripts to evaluate the different settings covered in the paper.

## Data
The datasets used in this work are publicly available:
* [SEN12MS](https://mediatum.ub.tum.de/1474000)
* [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files)

## Checkpoints
We provide checkpoints for ResNet50 and Swin-t transformer models trained with Sentinel-1/2 pairs from SEN12MS (self-supervised):
* [ResNet50](https://drive.google.com/file/d/1txqsNLUhIiQXRxflK_SMHn4y7Pp8o3mZ/view?usp=sharing)
* [Swin-t](https://drive.google.com/file/d/1e7QhC2-zoBFgb65yN1ADWBZLMShHOweY/view?usp=sharing)

## Code
This repository incorporates code from the following sources:
* [Data handling](https://github.com/lukasliebel/dfc2020_baseline)
* [SimCLR](https://github.com/sthalles/SimCLR)
* [Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL)
