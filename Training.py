import sys
import argparse
from typing import Dict, Any, Tuple, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.append('DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2')
from tacotron2.model import Tacotron2
from tacotron2.data_function import TextMelCollate
from tacotron2.loss_function import Tacotron2Loss

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path: str) -> Dataset:
    try:
        return load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        return sample['text'], sample['mel_spectrogram']

def create_dataloader(dataset: Dataset, batch_size: int, is_train: bool = True) -> DataLoader:
    custom_dataset = CustomDataset(dataset)
    collate_fn = TextMelCollate(n_frames_per_step=1)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn)

def load_model(model_path: str, device: torch.device) -> Tacotron2:
    model = Tacotron2().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    return model

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_thresh: float,
    scaler: GradScaler
) -> float:
    model.train()
    total_loss: float = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        inputs, targets = [x.to(device) for x in batch]

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

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

def main(args: argparse.Namespace) -> None:
    config: Dict[str, Any] = load_config(args.config)
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    dataset: Dataset = load_dataset(config['dataset_path'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader: DataLoader = create_dataloader(train_dataset, config['batch_size'])
    val_dataloader: DataLoader = create_dataloader(val_dataset, config['batch_size'], is_train=False)

    # Model setup
    model: nn.Module = load_model(config['pretrained_model_path'], device)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion: nn.Module = Tacotron2Loss()
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Mixed precision setup
    scaler = GradScaler()

    # Tensorboard setup
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(config['num_epochs']):
        train_loss: float = train_epoch(model, train_dataloader, optimizer, criterion, device, config['grad_clip_thresh'], scaler)
        val_loss: float = validate(model, val_dataloader, criterion, device)

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, f'{args.checkpoint_dir}/best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        save_checkpoint(model, optimizer, epoch, val_loss, f'{args.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth')

    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tacotron2 model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    args = parser.parse_args()
    main(args)
