"""
Conditional DDPM Pipeline for Image Generation

This module defines a custom `DiffusionPipeline` implementation that supports
class-conditional image generation using a DDPM model (e.g., UNet2DModel with class embedding).
The generation is performed by iteratively denoising from Gaussian noise using a trained scheduler.

Classes
-------
- DDPMConditionalPipeline: Implements the forward denoising process with label conditioning.
"""

from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class DDPMConditionalPipeline(DiffusionPipeline):
    """
    Conditional denoising pipeline for class-conditioned image generation using DDPM.

    Parameters
    ----------
    unet : torch.nn.Module
        The U-Net model with class-conditioning support.
    scheduler : diffusers.schedulers
        Scheduler implementing the denoising process (e.g. DDPMScheduler).

    Methods
    -------
    __call__(...)
        Generates images conditioned on the provided class labels.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        labels: torch.Tensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Run the diffusion process to generate images conditioned on labels.

        Parameters
        ----------
        labels : torch.Tensor
            Tensor of class labels to condition image generation.
        batch_size : int, optional
            Number of images to generate (default is 1).
        generator : torch.Generator or list, optional
            Random number generator for reproducibility.
        num_inference_steps : int, optional
            Number of denoising steps (default is 1000).
        output_type : str, optional
            Format of the returned images ('pil' or 'numpy').
        return_dict : bool, optional
            If True, returns ImagePipelineOutput; otherwise, returns a tuple.

        Returns
        -------
        Union[ImagePipelineOutput, Tuple]
            Generated images in the specified format.
        """
        # Sample Gaussian noise as starting point
        if isinstance(self.unet.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.sample_size,
                self.unet.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.sample_size,
            )

        image = randn_tensor(image_shape, generator=generator, device=self.device)

        # Set number of timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Predict the noise residual
            model_output = self.unet(image, t, labels).sample

            # Compute the denoised image (x_t -> x_{t-1})
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        # Post-process the image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
