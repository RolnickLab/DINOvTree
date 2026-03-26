import math
from typing import Any, Final

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(
    optimizer: Optimizer,
    base_lr: float,
    total_steps: int,
    warmup_epochs: int,
    niter_per_epoch: int,
    warmup_lr: float,
    min_lr: float,
) -> dict[str, Any]:
    """
    Builds a Cosine Learning Rate scheduler with a linear warmup phase.

    The scheduler transitions from a linear increase (warmup) to a cosine
    decay. The return format is designed for PyTorch Lightning's
    configure_optimizers.

    Args:
        optimizer: The wrapped optimizer.
        base_lr: The reference maximum learning rate.
        total_steps: Total number of optimization steps.
        warmup_epochs: Number of epochs dedicated to linear warmup.
        niter_per_epoch: Number of iterations (batches) per epoch.
        warmup_lr: Initial learning rate at the start of warmup.
        min_lr: The final learning rate at the end of the cosine decay.

    Returns:
        A dictionary containing the LambdaLR scheduler and Lightning
        configuration metadata.
    """
    warmup_steps: Final[int] = int(warmup_epochs * niter_per_epoch)
    cosine_steps: Final[int] = total_steps - warmup_steps

    def lr_lambda(current_step):
        """
        Calculates the learning rate scaling factor for a given step.

        Args:
            current_step: The current global optimization step.

        Returns:
            A multiplier for the base learning rate.
        """
        if current_step < warmup_steps:
            warmup_ratio = current_step / max(1, warmup_steps)
            return (warmup_lr / base_lr) + warmup_ratio * (1.0 - warmup_lr / base_lr)

        step_in_cosine = current_step - warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_cosine / cosine_steps))
        return (min_lr / base_lr) + cosine_decay * (1.0 - min_lr / base_lr)

    scheduler = {
        "scheduler": LambdaLR(optimizer, lr_lambda),
        "interval": "step",
        "frequency": 1,
        "name": "cosine_lr",
    }
    return scheduler
