from typing import List, Optional, Tuple, Union
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

import torch


class DDPMConditionalPipeline(DiffusionPipeline):
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
        # Sample gaussian noise to begin loop
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

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, labels).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
