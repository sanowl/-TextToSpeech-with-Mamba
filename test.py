import os
import logging
from typing import Dict, Any
import torchaudio.transforms as T
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
MAX_WAV_VALUE = 32768.0

# Load the LJSpeech dataset
def load_lj_speech() -> Dataset:
    try:
        return load_dataset("lj_speech", split="train")
    except Exception as e:
        logging.error(f"Failed to load LJSpeech dataset: {e}")
        raise

# Function to preprocess text
def preprocess_text(text: str) -> str:
    return text.lower()

# Function to preprocess audio
def preprocess_audio(audio: Dict[str, Any]) -> torch.Tensor:
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    waveform = waveform / MAX_WAV_VALUE

    transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        center=False
    )

    mel_spec = transform(waveform)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec

# Apply preprocessing to a single item
def preprocess_item(item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        item["text"] = preprocess_text(item["text"])
        item["mel_spectrogram"] = preprocess_audio(item["audio"])
        return item
    except Exception as e:
        logging.warning(f"Failed to preprocess item: {e}")
        return None

# Apply preprocessing to the dataset
def preprocess_dataset(dataset: Dataset, num_proc: int = 4) -> Dataset:
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(preprocess_item, item) for item in dataset]

        preprocessed_data = []
        for future in tqdm(as_completed(futures), total=len(dataset), desc="Preprocessing"):
            result = future.result()
            if result is not None:
                preprocessed_data.append(result)

    return Dataset.from_dict({
        key: [item[key] for item in preprocessed_data]
        for key in preprocessed_data[0].keys()
        if key not in ["audio", "file"]
    })

def main():
    output_path = "path_to_preprocessed_dataset"

    try:
        # Load dataset
        dataset = load_lj_speech()
        logging.info(f"Loaded dataset with {len(dataset)} items")

        # Preprocess dataset
        preprocessed_dataset = preprocess_dataset(dataset)
        logging.info(f"Preprocessed dataset with {len(preprocessed_dataset)} items")

        # Save the preprocessed dataset
        preprocessed_dataset.save_to_disk(output_path)
        logging.info(f"Dataset preprocessed and saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    main()
