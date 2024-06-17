import torchaudio.transforms as T
import torch
from datasets import load_dataset

# Load the LJSpeech dataset
dataset = load_dataset("lj_speech", split="train")

# Function to preprocess text (simple lowercasing for this example)
def preprocess_text(text):
    return text.lower()

# Function to preprocess audio
def preprocess_audio(audio):
    waveform = torch.tensor(audio["array"], dtype=torch.float32)  # Ensure waveform is float32
    sample_rate = audio["sampling_rate"]
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=80)
    mel_spec = transform(waveform)
    return mel_spec

# Apply preprocessing to the dataset
def preprocess_batch(batch):
    batch["text"] = [preprocess_text(text) for text in batch["text"]]
    batch["mel_spectrogram"] = [preprocess_audio(audio) for audio in batch["audio"]]
    return batch

# Apply the preprocessing to the dataset
dataset = dataset.map(preprocess_batch, remove_columns=["audio", "file"], batched=True)

# Save the preprocessed dataset
dataset.save_to_disk("path_to_preprocessed_dataset")

print("Dataset preprocessed and saved!")
