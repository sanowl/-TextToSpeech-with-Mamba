from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from scipy.io import wavfile
from torch import nn
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpeechParams:
    speed_factor: float = 1.0
    energy_factor: float = 1.0
    pitch_factor: float = 1.0

class TTSModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt")
        return self.model.generate(**inputs).mel_outputs.squeeze()

class Vocoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(mel_spectrogram.unsqueeze(0)).audio.squeeze()

class TextToSpeechPipeline:
    def __init__(self, tts_model_name: str, vocoder_name: str, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tts_model = TTSModel(tts_model_name).to(self.device).eval()
        self.vocoder = Vocoder(vocoder_name).to(self.device).eval()

    @torch.no_grad()
    def generate_speech(self, text: str, params: SpeechParams) -> Tuple[torch.Tensor, int]:
        mel_spectrogram = self.tts_model(text)
        modified_mel = self._modify_spectrogram(mel_spectrogram, params)
        waveform = self.vocoder(modified_mel)
        return waveform, self.tts_model.model.config.sampling_rate

    def _modify_spectrogram(self, mel_spec: torch.Tensor, params: SpeechParams) -> torch.Tensor:
        mel_spec = self._adjust_speed(mel_spec, params.speed_factor)
        mel_spec = self._adjust_energy(mel_spec, params.energy_factor)
        mel_spec = self._adjust_pitch(mel_spec, params.pitch_factor)
        return mel_spec

    @staticmethod
    def _adjust_speed(mel_spec: torch.Tensor, speed_factor: float) -> torch.Tensor:
        return (torchaudio.transforms.TimeStretch(n_freq=mel_spec.shape[0], hop_length=256)(mel_spec, speed_factor)
                if speed_factor != 1.0 else mel_spec)

    @staticmethod
    def _adjust_energy(mel_spec: torch.Tensor, energy_factor: float) -> torch.Tensor:
        return mel_spec * energy_factor

    @staticmethod
    def _adjust_pitch(mel_spec: torch.Tensor, pitch_factor: float) -> torch.Tensor:
        if pitch_factor == 1.0:
            return mel_spec
        freq_bins = mel_spec.shape[0]
        shift = int(freq_bins * (pitch_factor - 1))
        mel_spec = torch.roll(mel_spec, shifts=shift, dims=0)
        if shift > 0:
            mel_spec[:shift, :] = mel_spec[shift, :]
        else:
            mel_spec[shift:, :] = mel_spec[shift-1, :]
        return mel_spec

    @staticmethod
    def save_audio(waveform: torch.Tensor, sample_rate: int, filename: str) -> None:
        waveform_numpy = waveform.cpu().numpy()
        normalized = np.int16(waveform_numpy / np.max(np.abs(waveform_numpy)) * 32767)
        wavfile.write(filename, sample_rate, normalized)

class DatasetProcessor:
    def __init__(self, tts_pipeline: TextToSpeechPipeline, output_dir: str):
        self.tts_pipeline = tts_pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self, dataset_name: str, split: str, num_samples: int) -> List[Tuple[str, Path]]:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        processed_samples = []

        for sample in dataset.take(num_samples):
            text = sample["sentence"]
            logger.info(f"Processing: {text}")

            try:
                waveform, sample_rate = self.tts_pipeline.generate_speech(text, SpeechParams())
                output_file = self.output_dir / f"output_{sample['id']}.wav"
                self.tts_pipeline.save_audio(waveform, sample_rate, str(output_file))
                processed_samples.append((text, output_file))
            except Exception as e:
                logger.error(f"Error processing sample {sample['id']}: {e}")

        return processed_samples

def main():
    tts_pipeline = TextToSpeechPipeline("microsoft/speecht5_tts", "microsoft/speecht5_hifigan")

    # Example usage
    text = "Hello, how are you? This is an advanced text-to-speech system using Hugging Face models."
    params = SpeechParams(speed_factor=1.1, energy_factor=1.2, pitch_factor=0.95)

    try:
        waveform, sample_rate = tts_pipeline.generate_speech(text, params)
        output_file = Path("output_speech.wav")
        tts_pipeline.save_audio(waveform, sample_rate, str(output_file))
        logger.info(f"Generated speech saved as '{output_file}'")
    except Exception as e:
        logger.error(f"Error generating speech: {e}")

    # Process dataset
    processor = DatasetProcessor(tts_pipeline, "dataset_outputs")
    try:
        processed_samples = processor.process_dataset("common_voice", "en", num_samples=5)
        for text, file in processed_samples:
            logger.info(f"Processed '{text}' -> '{file}'")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")

if __name__ == "__main__":
    main()
