import os
import yaml
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as T


from pathlib import Path
from dataset import Flickr30kDataset
from torch.nn.utils.rnn import pad_sequence
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import corpus_bleu

TRANSFORM_REGISTRY = {
    "RandomResizedCrop": T.RandomResizedCrop,
    "Resize": T.Resize,
    "CenterCrop": T.CenterCrop,
    "HorizontalFlip": T.RandomHorizontalFlip,
    "ColorJitter": T.ColorJitter,
    "ToTensor": T.ToTensor,
    "Normalize": T.Normalize,
    "ConvertImageDtype": T.ConvertImageDtype,
}


class Collate():
    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Stack images
        imgs = torch.stack([item[0] for item in batch], dim=0)

        # Padding captions
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        # Raw captions
        raw_captions = [item[2] for item in batch]

        return imgs, captions, raw_captions
    
    
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def split_image_by_id(caption_df, train_ratio, val_ratio, test_ratio, seed=42):
    """A helper function that split images into train/val/test using images' ids instead by of row number to prevent data leakage

    Args:
        caption_df (_type_): _description_
        train_ratio (_type_): _description_
        val_ratio (_type_): _description_
        test_ratio (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    image_ids = caption_df["image_name"].unique()
    
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    
    train_ratio = 0.8 if not train_ratio else train_ratio
    val_ratio = 0.1 if not val_ratio else val_ratio
    
    total_image = len(image_ids)
    train_num = int(total_image * train_ratio)
    val_num = int(total_image * val_ratio)
    
    train_ids = image_ids[:train_num]
    val_ids = image_ids[train_num:train_num + val_num]
    
    train_df = caption_df[caption_df["image_name"].isin(train_ids)].reset_index(drop=True)
    val_df = caption_df[caption_df["image_name"].isin(val_ids)].reset_index(drop=True)
    
    split_log = f"Train (Ratio={train_ratio} | Num={train_num}); Val (Ratio={val_ratio} | Num={val_num})"
    
    # Create a test_df if test_ratio provided
    if test_ratio:    
        
        test_num = int(total_image * test_ratio)
        test_ids = image_ids[train_num + val_num:train_num + val_num + test_num]
        test_df = caption_df[caption_df["image_name"].isin(test_ids)].reset_index(drop=True)
    
        split_log += f"; Test (Ratio={test_ratio} | Num={test_num})"

        print(split_log)

        return train_df, val_df, test_df
    
    print(split_log)
    
    return train_df, val_df


def build_transforms(transform_cfg):
    transforms = []

    for t in transform_cfg:
        name = t["name"]
        params = {}
        
        for k, v in t.items():
            if k == "name":
                continue
            if name == "ConvertImageDtype" and k == "dtype":
                params[k] = torch.float32
            else:
                params[k] = v

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform: {name}")

        transforms.append(TRANSFORM_REGISTRY[name](**params))

    return T.Compose(transforms)
    

def get_loader(
    img_dir,
    captions_file,
    transform,
    vocab,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = Collate(pad_idx=pad_idx)
    train_transforms = build_transforms(transform)

    # Create train/val/test datasets
    train_dataset = Flickr30kDataset(img_dir, captions_file, vocab, train_transforms)

    # Create train loader
    data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return data_loader


def get_reference_captions(caption_df):
    referece_captions = (
        caption_df
        .groupby("image_name")[" comment"]
        .apply(list)
        .to_dict()
    )
    return referece_captions


def get_bleu_score(referece_dict, generated_dict):
    all_references = []
    all_generated = []
    REMOVE_TOKENS = {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}
    
    for image_id in referece_dict.keys():
        reference_sentences = []
        
        for caption in referece_dict[image_id]:
            tokens = caption.split(" ")
            reference_sentences.append(tokens)
        
        all_references.append(reference_sentences)

    for image_id in generated_dict.keys():
        tokens = generated_dict[image_id].split(" ")
        tokens = [token for token in tokens if token not in REMOVE_TOKENS]
        all_generated.append(tokens)

    bleu_score = corpus_bleu(all_references, all_generated)
    
    return bleu_score


def get_cider_score(reference_captions, generated_captions):
    all_references = {}
    all_generated = {}
    REMOVE_TOKENS = {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}
    
    for image_id in generated_captions:
        sentence = generated_captions[image_id]

        tokens = sentence.split()
        tokens = [t for t in tokens if t not in REMOVE_TOKENS]

        generated_captions[image_id] = " ".join(tokens)
    
    for image_id in reference_captions.keys():
        if image_id not in all_references.keys():
            all_references[image_id] = []
            all_generated[image_id] = []
            
        for caption in reference_captions[image_id]:
            all_references[image_id].append(caption)
            
        all_generated[image_id].append(generated_captions[image_id])
            
    cider = Cider()
    score, _ = cider.compute_score(all_references, all_generated)
    
    return score


def denormalize(img, mean, std):
    """Helper function to denormalize the image before visualize

    Args:
        img (_type_): _description_
        mean (_type_): _description_
        std (_type_): _description_

    Returns:
        _type_: _description_
    """
    mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.tensor(std, device=img.device).view(-1, 1, 1)
    return img * std + mean


def plot_image_with_caption(
    img,
    caption,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    """Helper function to visualize the image along with its caption.

    Args:
        img (_type_): _description_
        caption (_type_): _description_
        mean (list, optional): _description_. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): _description_. Defaults to [0.229, 0.224, 0.225].
    """
    # Ensure the dimension as plt expectation
    if torch.is_tensor(img):
        img = img.detach().cpu()
        img = denormalize(img, mean, std)
        
        if img.ndim == 4:
            img = img.squeeze(0)
        
        if img.ndim == 3:
            img = img.permute(1, 2, 0)

    _, ax = plt.subplots()

    ax.imshow(img)
    ax.set_axis_off()
    ax.text(0.5, -0.1, caption, ha='center', va='top', fontsize=10, wrap=True, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


def plot_loss(train_losses, val_losses, time_date, output_dir="graphs"):
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f"{time_date}_loss.png"), bbox_inches="tight")
    plt.close()
    
    
def save_cap_length_plot():
    pass


def save_checkpoint(
    save_dir,
    epoch,
    model,
    optimizer,
    scheduler=None,
    date_time=None,
    is_best=False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_best:
        prefix = "best"
    else:
        prefix = "latest"
        
    filename = f"{prefix}_checkpoint_{date_time}.pth"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_dir / filename)
    
    return save_dir / filename


def load_checkpoint(
    ckpt_path,
    model,
    optimizer,
    scheduler=None,
    device="cpu",
):
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    epoch = checkpoint.get("epoch", 0)
    
    return epoch
