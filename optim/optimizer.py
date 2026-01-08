import torch.optim as optim

def build_optimizer(model, cfg):
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg["name"].lower()

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            betas=opt_cfg.get("betas", (0.9, 0.999))
        )

    elif name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=opt_cfg.get("betas", (0.9, 0.999))
        )

    elif name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0)
        )

    else:
        raise ValueError(f"Optimizer not supported!")
