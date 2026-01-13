import os
import utils
import torch
import random
import argparse
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dataset import Vocabulary
from models.encoder import CNNEncoder
from models.decoder import LSTMDecoder
from models.model import EncoderDecoder
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler


def get_args():
    parser = argparse.ArgumentParser(
        description="Image Captioning Training"
    )
    
    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    
    # Epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    
    # Learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    
    # Resume
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
    )
    
    # Checkpoint
    parser.add_argument(
        "--ckpt",
        type=str,
    )
    
    # Teacher forcing ratio
    parser.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        default=0.5,
    )
    
    # Beam size
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
    )    
    
    # Max len
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
    )
    
    return parser.parse_args()


def override(cfg, args):
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        
    if args.device is not None:
        cfg["device"] = args.device
        
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
        
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
        
    if args.teacher_forcing_ratio is not None:
        cfg["training"]["teacher_forcing_ratio"] = args.teacher_forcing_ratio

    if args.beam_size is not None:
        cfg["training"]["beam_size"] = args.beam_size
        
    if args.max_len is not None:
        cfg["training"]["max_len"] = args.max_len

    return cfg


def main():
    # 1. Set up config
    DATA_CONFIG_DIR = "./config.yaml"
    config = utils.load_config(DATA_CONFIG_DIR)
    
    args = get_args()
    
    config = override(config, args)
    
    captions_file = config["data"]["captions_file"]
    img_dir = config["data"]["img_dir"]
    
    backbone = config["model"]["backbone"]
    embed_dim = config["model"]["embed_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    encoder_dim = config["model"]["encoder_dim"]
    
    save_dir = Path(config["training"]["save_dir"])
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    shuffle = config["training"]["shuffle"]
    pin_memory = config["training"]["pin_memory"]
    teacher_forcing_ratio = config["training"]["teacher_forcing_ratio"]
    beam_size = config["training"]["beam_size"]
    max_len = config["training"]["max_len"]
    
    device = config["device"]
    
    if device is not None:
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device("cpu")
    print(f"Running on {device}")
    
    # 2. Set up data
    df = pd.read_csv(captions_file, sep='|')
    
    vocabulary = Vocabulary(config["data"]["frequent_threshold"])
    vocabulary.build_vocabulary(df[' comment'].tolist())
    vocab_size = len(vocabulary)
    
    TRAIN_RATIO = config["training"]["train_ratio"]
    VAL_RATIO = config["training"]["val_ratio"]
    TEST_RATIO = config["training"]["test_ratio"]
    
    train_df, val_df, test_df = utils.split_image_by_id(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    train_loader = utils.get_loader(img_dir, train_df, config["data"]["transform"]["train"],
                                    vocabulary, batch_size, num_workers, shuffle, pin_memory)
    
    val_loader = utils.get_loader(img_dir, val_df, config["data"]["transform"]["train"],
                                  vocabulary, batch_size, num_workers, shuffle, pin_memory)
    
    test_loader = utils.get_loader(img_dir, test_df, config["data"]["transform"]["train"],
                                   vocabulary, batch_size, num_workers, shuffle, pin_memory)
    
    # 3.Build model
    encoder = CNNEncoder(backbone, encoder_dim).to(device)
    decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim, encoder_dim).to(device)
    model = EncoderDecoder(encoder, decoder, vocabulary).to(device)
    
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.stoi["<PAD>"])    
    scheduler = build_scheduler(optimizer, config)
    
    print(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler is None:
        print(f"Scheduler: None")
    else:
        print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Loss: {criterion.__class__.__name__}")
    
    # 4. Training
    date_time = datetime.now().strftime("%d%m%Y_%H%M")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    start_epoch = 1
    
    # Load old model if resume training
    if args.resume:
        ckpt_path = save_dir / args.ckpt
        start_epoch = utils.load_checkpoint(ckpt_path, model, optimizer, scheduler) + 1
        print(f"Continue training from {ckpt_path} at {start_epoch} epoch")
    
    for epoch in range(start_epoch, epochs+1):
        
        train_loss = train_epoch(model, train_loader, device, teacher_forcing_ratio, criterion, optimizer)
        train_losses.append(train_loss)
        
        val_loss = train_epoch(model, val_loader, device, teacher_forcing_ratio, criterion)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch}/{epochs}]: Training Loss {train_loss}, Validate Loss {val_loss}")
    
        if scheduler:
            scheduler.step()
            
        # Save the best checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            utils.save_checkpoint(save_dir, epoch, model, optimizer, date_time=date_time, is_best=is_best)
            best_val_loss = val_loss
            is_best = False
        
        # Save the latest checkpoint
        latest_ckpt = utils.save_checkpoint(save_dir, epoch, model, optimizer, date_time=date_time)
        
    utils.plot_loss(train_losses, val_losses, date_time)
        
    reference_captions = utils.get_reference_captions(test_df)
    bleu_1_score, bleu_4_score, cider_score = test_epoch(model, test_loader, reference_captions, max_len, beam_size, device)
    
    print(f"Testing result:")
    print(f"BLEU-1 Score: {bleu_1_score}")
    print(f"BLEU-4 Score: {bleu_4_score}")
    print(f"CIDEr Score: {cider_score}")
    
    try:
        print(f"Checkpoint saved at {latest_ckpt}")
    except:
        print(f"No checkpoint was saved in this training session")
    
    inference(model, test_loader, device)
    
    
def train_epoch(model, data_loader, device, teacher_forcing_ratio=0.5, criterion=None, optimizer=None):
    total_loss = 0

    # Train if optimizer is provided
    if optimizer:
        model.train()
        for imgs, captions, _ in tqdm(data_loader):
    
            imgs = imgs.to(device)
            captions = captions.to(device)       
            
            outputs = model(imgs, captions[:, :-1], teacher_forcing_ratio)
            
            loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:,1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
        
    # Otherwise validate
    else:
        model.eval()
        with torch.no_grad():
            for imgs, captions, _ in tqdm(data_loader):
        
                imgs = imgs.to(device)
                captions = captions.to(device)       
                
                outputs = model(imgs, captions[:, :-1], teacher_forcing_ratio)
                
                loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:,1:].reshape(-1))
                
                total_loss += loss.item()
            
    return total_loss/len(data_loader)


def test_epoch(model, test_loader, reference_captions, max_len, beam_size, device):
    generated_captions = {}
    
    model.eval()
    with torch.no_grad():
        for images, _, image_ids in tqdm(test_loader):
            images = images.to(device)
            
            generated_caption = model.generate(images, max_len=max_len, beam_size=beam_size)
            
            for idx, image_id in enumerate(image_ids):
                    
                generated_captions[image_id] = generated_caption[idx]
                
    # BLEU Score
    bleu_1_score = utils.get_bleu_score(reference_captions, generated_captions, weights=(1.0, 0, 0, 0))
    bleu_4_score = utils.get_bleu_score(reference_captions, generated_captions)
    
    # CIDEr Score
    cider_score = utils.get_cider_score(reference_captions, generated_captions)
                
    return bleu_1_score, bleu_4_score, cider_score
        
 
def inference(model, test_loader, device, test_num=5, save_fig=True):
    tested = 0
    model.eval()
        
    if save_fig:
        os.makedirs("./sample_result/", exist_ok=True)
    
    with torch.no_grad():
        for images, _, image_name in test_loader:
            images = images.to(device)
            
            idx = random.randint(0, images.size(0) - 1)
            image = images[idx].unsqueeze(0)
            img_id = image_name[idx].split(".")[0]
            
            generated_caption = model.generate(image)
            generated_caption = ' '.join(generated_caption)
            
            utils.plot_image_with_caption(image, generated_caption, img_id)
            
            tested += 1
            
            if tested > test_num:
                break
    
       
if __name__ == "__main__":
    main()
    