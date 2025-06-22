"""
Distributed Diffusion Models training script using Decentralized Federated Learning (DFL).

This module implements the decentralized training loop of Denoising Diffusion Probabilistic Models (DDPM)
across multiple nodes in a simulated federated topology. Each node trains locally and then aggregates weights
with its neighbors, without a central server. The module supports mixed precision training, checkpointing,
and generation of conditional image samples.

Classes
-------
- NodeInfo : Encapsulates the dataset and neighbor structure for a single node.
- ArgsTraining : Stores model, scheduler, dataloader, and optimizer state for training.
- DecentralizedFLDDPM : Handles the full decentralized training pipeline (local training, weight sharing, saving).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, notebook_launcher
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.core.config import TrainingConfig
from src.core.filesystem import reset
from src.core.pipeline import DDPMConditionalPipeline
from src.core.pipeline_utils import generate


@dataclass
class NodeInfo:
    """
    Encapsulates dataset and neighbors for a node in the decentralized topology.

    Attributes
    ----------
    dataset : Dataset
        Local dataset assigned to this node.
    neighbours : Optional[list[bool]]
        Boolean mask indicating which other nodes are neighbors (including self).
    """

    dataset: Dataset
    neighbours: Optional[list[bool]] = None


@dataclass
class ArgsTraining:
    """
    Stores all training components required for each node.

    Attributes
    ----------
    model : UNet2DModel
        The U-Net model used for DDPM.
    noise_scheduler : DDPMScheduler
        The noise scheduler used in training.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader with local training samples.
    lr_scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler.
    """

    model: UNet2DModel
    noise_scheduler: DDPMScheduler
    optimizer: torch.optim.Optimizer
    train_dataloader: torch.utils.data.DataLoader
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR


class DecentralizedFLDDPM:
    """
    Orchestrates decentralized training of DDPMs using Federated Learning principles.

    Parameters
    ----------
    config : TrainingConfig
        Configuration object containing all training parameters and directories.
    labels : List[int]
        List of unique class labels used for conditional generation.
    nodes_info : List[NodeInfo]
        List containing datasets and neighbor structure for each node.
    """

    def __init__(
        self, config: TrainingConfig, labels: List[int], nodes_info: List[NodeInfo]
    ):
        self.config = config
        self.labels = labels
        self.nodes_info = nodes_info
        self.log_file = os.path.join(self.config.output_dir, "log.txt")
        self.test_dir = os.path.join(self.config.output_dir, "samples")
        self.models_dir = os.path.join(self.config.output_dir, "models")

        self.global_noise_scheduler = DDPMScheduler()
        self.global_noise_scheduler.num_train_timesteps = 1000
        self.global_noise_scheduler.tensor_format = "pt"

    def prepare_args(self, accelerator: Accelerator, args: ArgsTraining):
        """Prepares model, dataloader, optimizer, and scheduler using Accelerate."""
        args.model, args.optimizer, args.train_dataloader, args.lr_scheduler = (
            accelerator.prepare(
                args.model.to(self.config.device),
                args.optimizer,
                args.train_dataloader,
                args.lr_scheduler,
            )
        )

    def train_model(
        self,
        accelerator: Accelerator,
        epoch: int,
        args: ArgsTraining,
        index: int,
        global_step: int,
    ) -> int:
        """
        Performs one epoch of training for a given node.

        Returns
        -------
        int
            Updated global training step.
        """
        progress_bar = tqdm(
            total=len(args.train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}, Model {index}")

        for _, batch in enumerate(args.train_dataloader):
            clean_images = batch[0] * 2 - 1
            class_labels = batch[1]
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                args.noise_scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            noisy_images = args.noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )

            with accelerator.accumulate(args.model):
                noise_pred = args.model(noisy_images, timesteps, class_labels).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(args.model.parameters(), 1.0)
                args.optimizer.step()
                args.lr_scheduler.step()
                args.optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": args.lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        return global_step

    def aggregate_weights(self, *args: ArgsTraining):
        """
        Aggregates model weights across neighbors for all nodes.
        Each node averages weights with its own and its neighbors'.
        """
        aggregated_weights = [{} for _ in args]
        num_neighbours = [0] * len(args)

        for i, node in enumerate(self.nodes_info):
            neighbours = node.neighbours
            num_neighbours[i] = sum(neighbours)

            for j, is_neighbour in enumerate(neighbours):
                if not is_neighbour:
                    continue
                for key, value in args[j].model.state_dict().items():
                    if key not in aggregated_weights[i]:
                        aggregated_weights[i][key] = value.clone().detach()
                    else:
                        aggregated_weights[i][key] += value.clone().detach()

        for i, arg in enumerate(args):
            if num_neighbours[i] > 0:
                for key in aggregated_weights[i]:
                    aggregated_weights[i][key] /= num_neighbours[i]
                arg.model.load_state_dict(aggregated_weights[i])

    def save(
        self,
        accelerator: Accelerator,
        epoch: int,
        global_step: int,
        *args: ArgsTraining,
    ):
        """
        Saves model checkpoints and generates image samples at scheduled epochs.
        """
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and (
            (epoch + 1) % self.config.save_epochs == 0
            or epoch == self.config.num_epochs - 1
        ):
            for i, _ in enumerate(args):
                pipeline = DDPMConditionalPipeline(
                    unet=accelerator.unwrap_model(args[i].model),
                    scheduler=args[i].noise_scheduler,
                )
                pipeline.save_pretrained(
                    os.path.join(self.models_dir, "model" + str(i + 1))
                )
                generate(
                    self.test_dir,
                    self.labels,
                    self.config.device,
                    epoch,
                    i + 1,
                    pipeline,
                )

            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"{epoch + 1} {global_step}")

    def train_loop(self, *args: ArgsTraining):
        """
        Executes the full training loop across all nodes and epochs.
        """
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        for arg in args:
            self.prepare_args(accelerator, arg)

        saved_epoch = 0
        global_step = 0
        if not self.config.overwrite_output_dir and os.path.isfile(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                line = f.read().split()
                saved_epoch = int(line[0])
                global_step = int(line[1])

        for epoch in range(saved_epoch, self.config.num_epochs):
            for i, _ in enumerate(args):
                global_step = self.train_model(
                    accelerator, epoch, args[i], i + 1, global_step
                )
            self.aggregate_weights(*args)
            self.save(accelerator, epoch, global_step, *args)

    def init_args(self, index: int) -> ArgsTraining:
        """
        Initializes the training arguments for a node (model, dataloader, scheduler).

        Parameters
        ----------
        index : int
            Index of the node in the topology.

        Returns
        -------
        ArgsTraining
            Prepared arguments for training.
        """
        train_dataloader = torch.utils.data.DataLoader(
            self.nodes_info[index].dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )

        if self.config.overwrite_output_dir or not os.path.isdir(self.models_dir):
            model = UNet2DModel(
                sample_size=self.config.image_size,
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(64, 128, 128),
                down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                num_class_embeds=len(self.labels),
            )
            noise_scheduler = DDPMScheduler()
            noise_scheduler.num_train_timesteps = 1000
            noise_scheduler.tensor_format = "pt"
        else:
            model_path = os.path.join(self.models_dir, f"model{index + 1}")
            model = UNet2DModel.from_pretrained(os.path.join(model_path, "unet"))
            noise_scheduler = DDPMScheduler.from_pretrained(
                os.path.join(model_path, "scheduler")
            )
            model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * self.config.num_epochs),
        )

        return ArgsTraining(
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
        )

    def run(self):
        """
        Prepares the output directory and launches the decentralized training process.
        """
        if self.config.overwrite_output_dir and os.path.isdir(self.config.output_dir):
            reset(self.config.output_dir)

        args = [self.init_args(i) for i, _ in enumerate(self.nodes_info)]
        notebook_launcher(
            self.train_loop, args, num_processes=self.config.num_processes
        )
