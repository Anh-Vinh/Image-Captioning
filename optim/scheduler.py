import math
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LambdaLR
)

def build_scheduler(optimizer, cfg):
    sch_cfg = cfg["training"].get("scheduler")
    if sch_cfg is None:
        return None

    name = sch_cfg["name"].lower()
    epochs = cfg["training"]["epochs"]

    if name == "step":
        return StepLR(
            optimizer,
            step_size=sch_cfg["step_size"],
            gamma=sch_cfg.get("gamma", 0.1)
        )

    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sch_cfg.get("min_lr", 0.0)
        )

    elif name == "cosine_warmup":
        warmup_epochs = sch_cfg.get("warmup_epochs", 0)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sch_cfg.get("factor", 0.1),
            patience=sch_cfg.get("patience", 3),
            min_lr=sch_cfg.get("min_lr", 1e-6)
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Scheduler not supported!")
