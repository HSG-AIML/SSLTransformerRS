import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import get_dataset_similarities, get_rank_statistics
from metrics import ClasswiseAccuracy, ClasswiseMultilabelMetrics, PixelwiseMetrics


def validate_all(model, val_loader, criterion, device, config, model_name, target_name):
    model.eval()
    pbar = tqdm(val_loader)

    # track performance
    epoch_losses = torch.Tensor()
    if target_name == "single-classification":
        metrics = ClasswiseAccuracy(config.num_classes)
    elif target_name == "multi-classification":
        sigmoid = torch.nn.Sigmoid()
        metrics = ClasswiseMultilabelMetrics(config.num_classes)
    elif target_name == "pixel-classification":
        metrics = PixelwiseMetrics(config.num_classes)

    with torch.no_grad():
        for idx, sample in enumerate(pbar):

            if "x" in sample.keys():
                if torch.isnan(sample["x"]).any():
                    # some s1 scenes are known to have NaNs...
                    continue
            else:
                if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                    # some s1 scenes are known to have NaNs...
                    continue

            if model_name == "baseline" or model_name == "swin-baseline":
                s1 = sample["s1"]
                s2 = sample["s2"]
                if config.s1_input_channels == 0:
                    # no data fusion
                    img = s2.to(device)
                elif config.s2_input_channels == 0:
                    img = s1.to(device)
                else:
                    # data fusion
                    img = torch.cat([s1, s2], dim=1).to(device)

            elif model_name == "normal-simclr":
                x = sample["x"]
                img = x.to(device)

            elif model_name == "moby":
                img = torch.cat([sample["s1"], sample["s2"]], dim=1).to(device)

            elif model_name in [
                "dual-baseline",
                "dual-swin-baseline",
                "alignment",
                "simclr",
                "swin-t",
                "shared-swin-t",
                "shared-swin-t-baseline",
            ]:
                s1 = sample["s1"].to(device)
                s2 = sample["s2"].to(device)
                img = {"s1": s1, "s2": s2}

            if target_name == "single-classification":
                y = sample[config.target].long().to(device)
            elif target_name == "multi-classification":
                y = sample[config.target].to(device)
            elif target_name == "pixel-classification":
                y = sample[config.target].squeeze().type(torch.LongTensor).to(device)

            y_hat = model(img)

            if target_name == "multi-classification":
                y_hat = sigmoid(y_hat)

            loss = criterion(y_hat, y)

            if target_name == "multi-classification":
                pred = y_hat.round()
            elif target_name == "single-classification":
                _, pred = torch.max(y_hat, dim=1)
            elif target_name == "pixel-classification":
                probas = F.softmax(y_hat, dim=1)
                pred = torch.argmax(probas, axis=1)

            epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
            metrics.add_batch(y, pred)

            pbar.set_description(f"Loss:{epoch_losses[-100:].mean():.4}")

        mean_loss = epoch_losses.mean()

        if target_name == "single-classification":
            val_stats = {
                "validation_loss": mean_loss.item(),
                "validation_average_accuracy": metrics.get_average_accuracy(),
                "validation_overall_accuracy": metrics.get_overall_accuracy(),
                **{
                    "validation_accuracy_" + k: v
                    for k, v in metrics.get_classwise_accuracy().items()
                },
            }

        elif target_name == "multi-classification":
            val_stats = {
                "validation_loss": mean_loss.item(),
                "validation_average_f1": metrics.get_average_f1(),
                "validation_overall_f1": metrics.get_overall_f1(),
                "validation_average_recall": metrics.get_average_recall(),
                "validation_overall_recall": metrics.get_overall_recall(),
                "validation_average_precision": metrics.get_average_precision(),
                "validation_overall_precision": metrics.get_overall_precision(),
                **{
                    "validation_f1_" + k: v
                    for k, v in metrics.get_classwise_f1().items()
                },
            }

        elif target_name == "pixel-classification":
            val_stats = {
                "validation_loss": mean_loss.item(),
                "validation_average_accuracy": metrics.get_average_accuracy(),
                **{
                    "validation_accuracy_" + k: v
                    for k, v in metrics.get_classwise_accuracy().items()
                },
            }

        return val_stats


def validate_alignment_backbone(
    model_s1, model_s2, dl, criterion, device, config, return_similarities=False
):
    model_s1.eval()
    model_s2.eval()
    all_s1_ses = []
    all_s2_ses = []
    pbar = tqdm(dl)
    # begin by encoding all scans
    print("Encoding images")
    for idx, sample in enumerate(pbar):
        s1_img = sample["s1"].to(device)
        s2_img = sample["s2"].to(device)

        with torch.no_grad():
            ses_s1 = model_s1(s1_img).cpu()
            ses_s2 = model_s2(s2_img).cpu()
            all_s1_ses.append(ses_s1)
            all_s2_ses.append(ses_s2)

    all_s1_ses = torch.cat(all_s1_ses)
    num_scans, _, _, _ = all_s1_ses.size()
    all_s2_ses = torch.cat(all_s2_ses)

    # now correlate encodings
    s1_b, s1_c, s1_h, s1_w = all_s1_ses.size()
    all_s1_ses = all_s1_ses.to(device)
    all_s2_ses = all_s2_ses.to(device)

    print("Calculating encoding similarities + statistics")
    similarities = get_dataset_similarities(all_s1_ses, all_s2_ses, device)
    # corrs = (F.conv2d(all_dxa_ses, all_mri_ses)/(mri_h*mri_w)).view(num_scans,num_scans,-1)
    rank_stats = get_rank_statistics(similarities)
    rank_stats = {"validation_" + k: v for k, v in rank_stats.items()}

    if not return_similarities:
        return rank_stats
    else:
        return rank_stats, similarities
