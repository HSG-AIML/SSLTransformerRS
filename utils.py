from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A


class AlbumentationsToTorchTransform:
    """Take a list of Albumentation transforms and apply them
    s.t. it is compatible with a Pytorch dataloader"""

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, x):
        x_t = self.augmentations(image=x)

        return x_t["image"]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomGrayscale(A.ToGray):
    def __init__(self, always_apply=False, p=0.5):
        super(A.ToGray, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if torch.rand(1).item() < self.p:
            img = np.repeat(img.mean(axis=2, keepdims=True), 12, axis=2)

        return img


def get_batch_corrrelations(scan_embeds_1, scan_embeds_2, device):
    """gets correlations between scan embeddings"""
    batch_size, channels, h, w = scan_embeds_2.shape

    scan_embeds_1 = F.normalize(scan_embeds_1, dim=1).to(device)
    scan_embeds_2 = F.normalize(scan_embeds_2, dim=1).to(device)
    correlation_maps = F.conv2d(scan_embeds_1, scan_embeds_2) / (h * w)
    return correlation_maps


class NCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch_similarities):
        ax1_softmaxes = F.softmax(batch_similarities / self.temperature, dim=1)
        ax2_softmaxes = F.softmax(batch_similarities / self.temperature, dim=0)
        softmax_scores = torch.cat(
            (-ax1_softmaxes.diag().log(), -ax2_softmaxes.diag().log())
        )
        loss = softmax_scores.mean()
        return loss


def normalise_channels(scan_img, eps=1e-5):
    # normalize each channel
    scan_min = scan_img.flatten(start_dim=-2).min(dim=-1)[0][:, None, None]
    scan_max = scan_img.flatten(start_dim=-2).max(dim=-1)[0][:, None, None]
    return (scan_img - scan_min) / (scan_max - scan_min + eps)


def save_checkpoint_single_model(
    model, optimiser, val_stats, epochs, save_weights_path
):

    print(f"==> Saving Model Weights to {save_weights_path}")
    state = {
        "model_weights": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "val_stats": val_stats,
        "epochs": epochs,
    }
    # if not os.path.isdir(save_weights_path):
    #    os.mkdir(save_weights_path)
    # previous_checkpoints = glob.glob(save_weights_path + '/ckpt*.pt', recursive=True)
    torch.save(state, save_weights_path)  # + '/ckpt' + str(epochs) + '.pt')
    # for previous_checkpoint in previous_checkpoints:
    #    os.remove(previous_checkpoint)
    return


def save_checkpoint(
    s1_model, s2_model, optimiser, val_stats, epochs, save_weights_path
):

    print(f"==> Saving Model Weights to {save_weights_path}")
    state = {
        "s1_model_weights": s1_model.state_dict(),
        "s2_model_weights": s2_model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "val_stats": val_stats,
        "epochs": epochs,
    }
    # if not os.path.isdir(save_weights_path):
    #    os.mkdir(save_weights_path)
    # previous_checkpoints = glob.glob(save_weights_path + '/ckpt*.pt', recursive=True)
    torch.save(state, save_weights_path)  # + '/ckpt' + str(epochs) + '.pt')
    # for previous_checkpoint in previous_checkpoints:
    #    os.remove(previous_checkpoint)
    return


def get_rank_statistics(similarities_matrix):
    sorted_similarities_values, sorted_similarities_idxs = similarities_matrix.sort(
        dim=1, descending=True
    )
    ranks = []
    for idx, row in enumerate(tqdm(sorted_similarities_idxs)):
        rank = torch.where(row == idx)[0][0]
        ranks.append(rank.cpu())
    ranks = np.array(ranks)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    top_10 = np.sum(ranks < 10) / len(ranks)
    top_5 = np.sum(ranks < 5) / len(ranks)
    top_1 = np.sum(ranks < 1) / len(ranks)

    ranks_stats = {
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "top_10": top_10,
        "top_5": top_5,
        "top_1": top_1,
    }

    return ranks_stats


def get_dataset_similarities(scan_embeds_1, scan_embeds_2, device, batch_size=50):
    """Gets similarities for entire dataset.
    Splits job into batches to reduce GPU memory"""
    ds_size, channels, h, w = scan_embeds_2.shape
    ds_similarities = torch.zeros(ds_size, ds_size)

    for batch_1_start_idx in tqdm(range(0, ds_size, batch_size)):
        for batch_2_start_idx in range(0, ds_size, batch_size):

            batch_1_end_idx = batch_1_start_idx + batch_size
            batch_2_end_idx = batch_2_start_idx + batch_size
            if batch_2_end_idx >= ds_size:
                batch_2_end_idx = ds_size
            if batch_1_end_idx >= ds_size:
                batch_1_end_idx = ds_size

            correlations = get_batch_corrrelations(
                scan_embeds_1[batch_1_start_idx:batch_1_end_idx],
                scan_embeds_2[batch_2_start_idx:batch_2_end_idx],
                device,
            )
            similarities, _ = torch.max(correlations.flatten(start_dim=2), dim=-1)
            ds_similarities[
                batch_1_start_idx:batch_1_end_idx, batch_2_start_idx:batch_2_end_idx
            ] = similarities
    return ds_similarities


class AverageMeter(object):
    """Computes and stores the average and current values"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def multi_acc(pred, label):
    """compute pixel-wise accuracy across a batch"""
    _, tags = torch.max(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


def class_wise_acc(pred, label, results, num_classes=10):
    """add number of correctly classified pixels and total number of pixels
    for each class to `results`"""
    _, tags = torch.max(pred, dim=1)

    for l in range(num_classes):
        if label[label == l].numel() == 0:
            continue
        else:
            corrects = (tags[label == l] == label[label == l]).float()
            results[str(l) + "_correct"] += corrects.sum()
            results[str(l) + "_total"] += corrects.numel()
            # acc = acc * 100
            # results[str(l)].extend(corrects.detach().cpu().numpy().tolist())

    return results


def class_wise_acc_per_img(pred, label, num_classes=10):
    """return class wise accuracy independently for each img in the batch
    assumes pred and label of dim bxnum_classesxhxw and bx1xhxw"""

    _, tags = torch.max(pred, dim=1)
    batch_size = pred.shape[0]

    results = []
    for b in range(batch_size):
        img_results = {}
        for l in range(num_classes):
            if label[b][label[b] == l].numel() == 0:
                # this class is not present in the current image
                continue
            else:
                corrects = (tags[b][label[b] == l] == label[b][label[b] == l]).float()
                img_results[str(l)] = (corrects.sum() / corrects.numel()).item() * 100

        results.append(img_results)

    return results


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dotdictify(d):
    """recursively wrap a dictionary and
    all the dictionaries that it contains
    with the dotdict class
    """
    d = dotdict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dotdictify(v)
    return d
