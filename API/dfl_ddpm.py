"""Distributed Diffusion Models training script """

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from accelerate import Accelerator, notebook_launcher
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from API.pipeline import DDPMConditionalPipeline
from API.utils import TrainingConfig, generate, reset


@dataclass
class NodeInfo:
    dataset: Dataset
    neighbours: Optional[list[bool]] = None


@dataclass
class ArgsTraining:
    model: UNet2DModel
    noise_scheduler: DDPMScheduler
    optimizer: torch.optim.Optimizer
    train_dataloader: torch.utils.data.DataLoader
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR


class DecentralizedFLDDPM:
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
        progress_bar = tqdm(
            total=len(args.train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}, Model {index}")

        for _, batch in enumerate(args.train_dataloader):
            clean_images = batch[0] * 2 - 1
            class_labels = batch[1]

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                args.noise_scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = args.noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )

            with accelerator.accumulate(args.model):
                # Predict the noise residual
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
        # Initialize an empty dictionary to store the aggregated weights
        aggregated_weights = [{} for _ in enumerate(args)]
        num_neighbours = [0] * len(args)

        # Iterate over each ArgsTraining object
        for i, node in enumerate(self.nodes_info):
            neighbours = node.neighbours  # Access neighbours only once
            num_neighbours[i] = sum(neighbours)  # Count active neighbours directly

            # Iterate over the neighbours, including self
            for j, is_neighbour in enumerate(neighbours):
                if not is_neighbour:
                    continue

                # Accumulate weights for neighbours
                for key, value in args[j].model.state_dict().items():
                    if key not in aggregated_weights[i]:
                        aggregated_weights[i][key] = value.clone().detach()
                    else:
                        aggregated_weights[i][key] += value.clone().detach()

        # Update local models
        for i, arg in enumerate(args):
            if num_neighbours[i] > 0:
                for key in aggregated_weights[i]:
                    aggregated_weights[i][key] /= num_neighbours[
                        i
                    ]  # Compute the average
                arg.model.load_state_dict(aggregated_weights[i])

    def save(
        self,
        accelerator: Accelerator,
        epoch: int,
        global_step: int,
        *args: ArgsTraining,
    ):
        # After each epoch optionally sample some demo images and save the model
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
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # Prepare everything
        for arg in args:
            self.prepare_args(accelerator, arg)

        saved_epoch = 0
        global_step = 0
        if not self.config.overwrite_output_dir and os.path.isfile(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                line = f.read().split()
                saved_epoch = int(line[0])
                global_step = int(line[1])

        # Train the model
        for epoch in range(saved_epoch, self.config.num_epochs):
            for i, _ in enumerate(args):
                global_step = self.train_model(
                    accelerator, epoch, args[i], i + 1, global_step
                )
            self.aggregate_weights(*args)
            self.save(accelerator, epoch, global_step, *args)

    def init_args(self, index: int) -> ArgsTraining:
        train_dataloader = torch.utils.data.DataLoader(
            self.nodes_info[index].dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )

        model, noise_scheduler = None, None

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
        if self.config.overwrite_output_dir and os.path.isdir(self.config.output_dir):
            reset(self.config.output_dir)

        args = [self.init_args(i) for i, _ in enumerate(self.nodes_info)]

        notebook_launcher(
            self.train_loop, args, num_processes=self.config.num_processes
        )
