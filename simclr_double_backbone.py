import logging
import os
import sys

import wandb
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from simclr_utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLRDoubleBackbone(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.use_logging = self.args.use_logging
        self.run_name = self.args.run_name
        self.writer = SummaryWriter()
        if self.use_logging:
            logging.basicConfig(
                filename=os.path.join(self.writer.log_dir, "training.log"),
                level=logging.DEBUG,
            )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(self.args.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        # scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        if self.use_logging:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc1_per_logging = []
        acc5_per_logging = []
        loss_per_logging = []

        for epoch_counter in range(self.args.epochs):
            pbar = tqdm(train_loader)
            for sample in pbar:

                # s1 = sample["s1"] # use both Sentinel-1 channels
                # s2 = sample["s2"][:, [4,3]] # use rg channels of Sentinel-2

                if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                    # some s1 scenes in sen12ms are known to have NaNs...
                    continue

                s1 = sample["s1"].to(self.args.device)
                s2 = sample["s2"].to(self.args.device)

                # model processes s1 and s2 data through different backbones
                images = {"s1": s1, "s2": s2}

                # with autocast(enabled=self.args.fp16_precision):
                feature_dict = self.model(images)

                features = torch.cat([feature_dict["s1"], feature_dict["s2"]])
                logits, labels = self.info_nce_loss(features)

                loss = self.criterion(logits, labels)

                if torch.isnan(loss):
                    print("Loss is nan:", loss)
                    return sample

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                acc1_per_logging.append(top1[0].item())
                acc5_per_logging.append(top5[0].item())
                loss_per_logging.append(loss.item())

                if n_iter % self.args.log_every_n_steps == 0:
                    # if n_iter == 0:
                    #    continue
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    mean_top1 = np.mean(acc1_per_logging)
                    mean_top5 = np.mean(acc5_per_logging)
                    mean_loss = np.mean(loss_per_logging)

                    if self.use_logging:
                        self.writer.add_scalar("loss", mean_loss, global_step=n_iter)
                        self.writer.add_scalar(
                            "acc/top1", mean_top1, global_step=n_iter
                        )
                        self.writer.add_scalar(
                            "acc/top5", mean_top5, global_step=n_iter
                        )
                        self.writer.add_scalar(
                            "learning_rate",
                            self.scheduler.get_lr()[0],
                            global_step=n_iter,
                        )

                    wandb.log(
                        {
                            "loss": mean_loss,
                            "acc/top1": mean_top1,
                            "acc/top5": mean_top5,
                            "learning_rate": self.scheduler.get_lr()[0],
                            "epoch": epoch_counter,
                        },
                        step=n_iter,
                    )

                    acc1_per_logging = []
                    acc5_per_logging = []
                    mean_loss = []

                # n_iter += 1
                n_iter += s1.shape[
                    0
                ]  # count the number of processed samples (i.e. batch_size * steps)
                pbar.set_description(
                    f"Epoch:{epoch_counter}, Step:{n_iter}, Loss:{np.mean(loss_per_logging[-100:]):.4}"
                )  # "{epoch_accuracy[-100:].mean():.4}")

            if epoch_counter % 25 == 0:
                print("Saving checkpoint for epoch:", epoch_counter)
                checkpoint_name = (
                    "checkpoints/simclr-double-backbone-"
                    + str(self.run_name)
                    + "-epoch"
                    + str(epoch_counter)
                    + ".pth"
                )
                save_checkpoint(
                    {
                        "epoch": epoch_counter,
                        "arch": self.args.arch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=checkpoint_name,
                )

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            if self.use_logging:
                logging.debug(
                    f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
                )

        if self.use_logging:
            logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = (
            "checkpoints/simclr-double-backbone-"
            + str(self.run_name)
            + "-epoch"
            + str(epoch_counter)
            + ".pth"
        )
        save_checkpoint(
            {
                "epoch": epoch_counter,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=checkpoint_name,
        )
        if self.use_logging:
            logging.info(
                f"Model checkpoint and metadata has been saved at {self.writer.log_dir}."
            )
