import sys
import argparse
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path

sys.path.append('DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2')
from tacotron2.model import Tacotron2
from tacotron2.data_function import TextMelCollate
from tacotron2.loss_function import Tacotron2Loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    dataset_path: str
    pretrained_model_path: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    grad_clip_thresh: float
    gradient_accumulation_steps: int
    early_stopping_patience: int
    save_interval: int

def load_config(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def load_dataset(dataset_path: str) -> Dataset:
    try:
        return load_from_disk(dataset_path)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        return sample['text'], sample['mel_spectrogram']

def create_dataloader(dataset: Dataset, batch_size: int, is_train: bool = True, num_workers: int = 4) -> DataLoader:
    custom_dataset = CustomDataset(dataset)
    collate_fn = TextMelCollate(n_frames_per_step=1)
    sampler = DistributedSampler(custom_dataset) if torch.distributed.is_initialized() else None
    return DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=(is_train and sampler is None),
        collate_fn=collate_fn,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )

def load_model(model_path: str, device: torch.device) -> Tacotron2:
    try:
        model = Tacotron2().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_thresh: float,
    scaler: GradScaler,
    gradient_accumulation_steps: int = 1
) -> float:
    model.train()
    total_loss: float = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for i, batch in enumerate(progress_bar):
        inputs, targets = [x.to(device) for x in batch]

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    return total_loss / len(dataloader)

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss: float = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = [x.to(device) for x in batch]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logger.info(f"Checkpoint saved to {path}")

def setup_training(config: TrainingConfig, device: torch.device) -> Tuple[
    nn.Module, torch.optim.Optimizer, nn.Module, ReduceLROnPlateau, GradScaler
]:
    model = load_model(config.pretrained_model_path, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = Tacotron2Loss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    scaler = GradScaler()
    return model, optimizer, criterion, scheduler, scaler

def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    config: TrainingConfig,
    device: torch.device,
    writer: SummaryWriter,
    checkpoint_dir: str
) -> None:
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(config.num_epochs):
        train_loss: float = train_epoch(
            model, train_dataloader, optimizer, criterion, device,
            config.grad_clip_thresh, scaler, config.gradient_accumulation_steps
        )
        val_loss: float = validate(model, val_dataloader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, f'{checkpoint_dir}/best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth')

    writer.close()
    logger.info("Training completed!")

def main(args: argparse.Namespace) -> None:
    config: TrainingConfig = load_config(args.config)
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # Data loading
    dataset: Dataset = load_dataset(config.dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader: DataLoader = create_dataloader(train_dataset, config.batch_size)
    val_dataloader: DataLoader = create_dataloader(val_dataset, config.batch_size, is_train=False)

    # Model setup
    model, optimizer, criterion, scheduler, scaler = setup_training(config, device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Tensorboard setup
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Start training
    train(
        model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, scaler,
        config, device, writer, str(checkpoint_dir)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tacotron2 model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    args = parser.parse_args()
    main(args)
