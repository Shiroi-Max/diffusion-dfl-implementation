"""
Image Generation Utilities for DDPM in Decentralized Federated Learning

This module provides functions to generate and save image samples from a trained
DDPMConditionalPipeline during training. It supports organizing outputs by epoch
and node index.

Functions
---------
- generate(...): Runs the DDPM pipeline to generate images and saves them in a structured layout.
- sample(...): Saves a given image grid to the appropriate directory structure.
"""

import os
from typing import List

import torch
from diffusers.utils import make_image_grid
from PIL.Image import Image

from src.core.pipeline import DDPMConditionalPipeline


def generate(
    root: str,
    labels: List[int],
    device: str,
    epoch: int,
    index: int,
    pipeline: DDPMConditionalPipeline,
):
    """
    Generate and save image samples using the provided DDPM pipeline.

    Parameters
    ----------
    root : str
        Base directory to save the samples.
    labels : List[int]
        List of class labels to condition generation on.
    device : str
        Device to run the generation on (e.g. "cuda" or "cpu").
    epoch : int
        Current training epoch, used for output folder structure.
    index : int
        Node index (1-based) used to distinguish between models in multi-node setups.
    pipeline : DDPMConditionalPipeline
        The diffusion model pipeline used to generate images.
    """
    y = torch.tensor([label for label in labels]).flatten().to(device)
    images = pipeline(y, y.size(0)).images

    grid = make_image_grid(images, len(labels), 1)

    sample(
        root,
        grid,
        epoch + 1,
        index,
        str(labels[0]) + "-" + str(labels[-1]),
    )


def sample(root: str, grid: Image, epoch: int, index: int, name: str):
    """
    Save the image grid to a structured directory based on model index and epoch.

    Parameters
    ----------
    root : str
        Base directory to save the sample.
    grid : PIL.Image.Image
        Image grid to save.
    epoch : int
        Current epoch (used in folder naming).
    index : int
        Model or node index (used in folder naming).
    name : str
        Base filename for the saved image.
    """
    os.makedirs(root, exist_ok=True)
    model_dir = os.path.join(root, f"model{index}")
    os.makedirs(model_dir, exist_ok=True)
    epoch_dir = os.path.join(model_dir, f"{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)

    grid.save(os.path.join(epoch_dir, name + "-sample.png"))
