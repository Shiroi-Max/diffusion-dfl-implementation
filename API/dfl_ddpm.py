from dataclasses import dataclass
from torch.utils.data import Dataset
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
from typing import Optional, List
from API.pipeline import DDPMConditionalPipeline
from API.utils import TrainingConfig
from PIL.Image import Image

import torch, os, shutil
import torch.nn.functional as F


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


config = None
log_file = None
test_dir = None
models_dir = None
labels = None
nodes_info = None

global_noise_scheduler = DDPMScheduler()
global_noise_scheduler.num_train_timesteps = 1000
global_noise_scheduler.tensor_format = "pt"


def reset(path: str):
    for filename in os.scandir(path):
        try:
            if os.path.isfile(filename) or os.path.islink(filename):
                os.unlink(filename)
            elif os.path.isdir(filename):
                print("Deleting directory %s" % filename)
                shutil.rmtree(filename)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (filename, e))
    os.rmdir(path)


def sample(grid: Image, epoch: int, index: int, name: str):
    os.makedirs(test_dir, exist_ok=True)
    model_dir = os.path.join(test_dir, f"model{index}")
    os.makedirs(model_dir, exist_ok=True)
    epoch_dir = os.path.join(model_dir, f"{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)

    grid.save(os.path.join(epoch_dir, name + "-sample.png"))


def generate(
    epoch: int,
    index: int,
    pipeline: DDPMConditionalPipeline,
):
    y = torch.tensor([label for label in labels]).flatten().to(config.device)
    images = pipeline(y, y.size(0)).images

    grid = make_image_grid(images, len(labels), 1)

    sample(grid, epoch + 1, index, str(labels[0]) + "-" + str(labels[-1]))


def prepareArgs(accelerator: Accelerator, args: ArgsTraining):
    args.model, args.optimizer, args.train_dataloader, args.lr_scheduler = (
        accelerator.prepare(
            args.model.to(config.device),
            args.optimizer,
            args.train_dataloader,
            args.lr_scheduler,
        )
    )


def train_model(
    accelerator: Accelerator,
    epoch: int,
    args: ArgsTraining,
    index: int,
    global_step: int,
) -> int:
    progress_bar = tqdm(
        total=len(args.train_dataloader), disable=not accelerator.is_local_main_process
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
        noisy_images = args.noise_scheduler.add_noise(clean_images, noise, timesteps)

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


def aggregateWeights(*args: ArgsTraining):
    # Initialize an empty dictionary to store the aggregated weights
    aggregated_weights = [{} for _ in range(len(args))]
    num_neighbours = [0] * len(args)

    # Iterate over each ArgsTraining object
    for i in range(len(args)):
        neighbours = nodes_info[i].neighbours
        # Iterate over each neighbour (also himself)
        for j in range(len(neighbours)):
            if neighbours[j] == False:
                continue
            num_neighbours[i] += 1
            # Iterate over the model's state_dict() and accumulate the weights
            for key, value in args[j].model.state_dict().items():
                if key not in aggregated_weights[i]:
                    aggregated_weights[i][key] = value.clone().detach()
                else:
                    aggregated_weights[i][key] += value.clone().detach()

        # Divide the accumulated weights by the number of neighbours
        for key in aggregated_weights[i]:
            aggregated_weights[i][key] /= (
                num_neighbours[i] if num_neighbours[i] > 0 else 1
            )

    # Update local models
    for i in range(len(args)):
        args[i].model.load_state_dict(aggregated_weights[i])


def save(accelerator: Accelerator, epoch: int, global_step: int, *args: ArgsTraining):
    # After each epoch you optionally sample some demo images and save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and (
        (epoch + 1) % config.save_epochs == 0 or epoch == config.num_epochs - 1
    ):
        for i in range(len(args)):
            pipeline = DDPMConditionalPipeline(
                unet=accelerator.unwrap_model(args[i].model),
                scheduler=args[i].noise_scheduler,
            )
            pipeline.save_pretrained(os.path.join(models_dir, "model" + str(i + 1)))
            generate(epoch, i + 1, pipeline)

        f = open(log_file, "w")
        f.write(f"{epoch + 1} {global_step}")
        f.close()


def train_loop(*args: ArgsTraining):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    for arg in args:
        prepareArgs(accelerator, arg)

    saved_epoch = 0
    global_step = 0
    if not config.overwrite_output_dir and os.path.isfile(log_file):
        f = open(log_file, "r")
        line = f.read().split()
        saved_epoch = int(line[0])
        global_step = int(line[1])
        f.close()

    # Train the model
    for epoch in range(saved_epoch, config.num_epochs):
        for i in range(len(args)):
            global_step = train_model(accelerator, epoch, args[i], i + 1, global_step)
        aggregateWeights(*args)
        save(accelerator, epoch, global_step, *args)


def initArgs(index: int) -> ArgsTraining:
    train_dataloader = torch.utils.data.DataLoader(
        nodes_info[index].dataset, batch_size=config.train_batch_size, shuffle=True
    )

    model, noise_scheduler = None, None

    if config.overwrite_output_dir or not os.path.isdir(models_dir):
        model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 128),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            num_class_embeds=len(labels),
        )
        noise_scheduler = DDPMScheduler()
        noise_scheduler.num_train_timesteps = 1000
        noise_scheduler.tensor_format = "pt"
    else:
        model_path = os.path.join(models_dir, f"model{index + 1}")
        model = UNet2DModel.from_pretrained(os.path.join(model_path, "unet"))
        noise_scheduler = DDPMScheduler.from_pretrained(
            os.path.join(model_path, "scheduler")
        )
        model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    return ArgsTraining(
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
    )


def run(_config: TrainingConfig, _labels: List[int], _nodes_info: List[NodeInfo]):
    global config, log_file, test_dir, models_dir, labels, nodes_info

    config = _config
    log_file = os.path.join(config.output_dir, "log.txt")
    test_dir = os.path.join(config.output_dir, "samples")
    models_dir = os.path.join(config.output_dir, "models")

    if config.overwrite_output_dir and os.path.isdir(config.output_dir):
        reset(config.output_dir)

    labels = _labels
    nodes_info = _nodes_info

    args = [initArgs(i) for i in range(len(nodes_info))]

    notebook_launcher(train_loop, args, num_processes=config.num_processes)
