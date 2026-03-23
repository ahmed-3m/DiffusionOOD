import torch
import torch.nn as nn
import diffusers
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConditionalUNet(nn.Module):
    """UNet2DModel with a binary class-conditioning head for OOD detection."""
    
    def __init__(
        self,
        sample_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        layers_per_block: int = 2,
        block_out_channels: tuple = (128, 256, 256, 256),
        down_block_types: tuple = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: tuple = (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds: int = 2,
    ):
        super().__init__()
        
        self.unet = diffusers.UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
        )
        
        self.num_class_embeds = num_class_embeds
        self.sample_size = sample_size
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted noise [B, C, H, W]."""
        return self.unet(sample, timestep, class_labels=class_labels).sample
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config) -> ConditionalUNet:
    """Instantiate a ConditionalUNet from a ModelConfig dataclass."""
    model = ConditionalUNet(
        sample_size=config.sample_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        num_class_embeds=config.num_class_embeds,
    )
    logger.info(f"Model: {model.get_num_params() / 1e6:.2f}M parameters")
    return model


def generate_model_card(
    config,
    metrics: dict,
    training_time: str = "",
) -> str:
    """Return a HuggingFace README card string for the trained model."""

    def _fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    card = f"""---
license: mit
tags:
  - diffusion
  - ood-detection
  - cifar10
  - conditional-generation
---

# Diffusion Classifier for OOD Detection

Binary conditional diffusion model trained on CIFAR-10 for out-of-distribution detection.

## Model Details

- **Architecture**: UNet2DModel with binary class conditioning
- **Classes**: {config.num_class_embeds} (c=0: ID, c=1: OOD proxy)
- **Input size**: {config.sample_size}x{config.sample_size}
- **Timesteps**: {config.num_train_timesteps}
- **Beta schedule**: {config.beta_schedule}

## Training

- **Dataset**: CIFAR-10 (airplane vs rest)
- **Duration**: {training_time}

## Metrics

| Metric | Value |
|--------|-------|
| AUROC  | {_fmt(metrics.get('auroc', 0.0))} |
| FPR@95 | {_fmt(metrics.get('fpr95', 1.0))} |
| AUPR   | {_fmt(metrics.get('aupr', 0.0))} |

## Usage

```python
from src.lightning_module import DiffusionClassifierOOD

model = DiffusionClassifierOOD.load_from_checkpoint("path/to/checkpoint.ckpt")
scores, predictions = model.score_images(images)
```
"""
    return card
