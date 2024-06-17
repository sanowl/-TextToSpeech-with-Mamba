import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import sys
sys.path.append('DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2')
from tacotron2.model import Tacotron2
from tacotron2.data_function import TextMelCollate
from tacotron2.loss_function import Tacotron2Loss

# Load the preprocessed dataset
dataset = load_from_disk("path_to_preprocessed_dataset")

# Custom Dataset class to load text and mel spectrogram pairs
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample['text'], sample['mel_spectrogram']

custom_dataset = CustomDataset(dataset)
collate_fn = TextMelCollate(n_frames_per_step=1)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Load Tacotron2 model
tacotron2 = Tacotron2()
tacotron2.load_state_dict(torch.load('nvidia_tacotron2.pth')['state_dict'])
tacotron2 = tacotron2.train()

# Define optimizer and loss function
optimizer = torch.optim.AdamW(tacotron2.parameters(), lr=1e-4)
criterion = Tacotron2Loss()

# Training loop
for epoch in range(10):  # Train for 10 epochs
    tacotron2.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Extract inputs and targets from the batch
        inputs, targets = batch
        
        # Forward pass
        outputs = tacotron2(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': tacotron2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch + 1}.pth')

print("Training completed!")
