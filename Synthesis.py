import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
from transformers import AutoProcessor, AutoModel, AutoConfig
from datasets import load_dataset
from scipy.io.wavfile import write

class TextToSpeechPipeline:
    def __init__(self, model_name: str, vocoder_name: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load vocoder
        self.vocoder = AutoModel.from_pretrained(vocoder_name).to(self.device)

        self.model.eval()
        self.vocoder.eval()

    @torch.no_grad()
    def generate_speech(self, text: str,
                        speed_factor: float = 1.0,
                        energy_factor: float = 1.0,
                        pitch_factor: float = 1.0) -> Tuple[torch.Tensor, int]:
        # Preprocess the text input
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        # Generate mel spectrogram
        outputs = self.model.generate(**inputs)

        mel_spectrogram = outputs.mel_outputs.squeeze()

        # Apply audio modifications
        mel_spectrogram = self._modify_spectrogram(mel_spectrogram, speed_factor, energy_factor, pitch_factor)

        # Convert mel spectrogram to audio waveform
        waveform = self.vocoder(mel_spectrogram.unsqueeze(0)).audio.squeeze()

        return waveform, self.model.config.sampling_rate

    def _modify_spectrogram(self, mel_spec: torch.Tensor,
                            speed_factor: float,
                            energy_factor: float,
                            pitch_factor: float) -> torch.Tensor:
        # Time stretching (speed modification)
        if speed_factor != 1.0:
            mel_spec = torchaudio.transforms.TimeStretch(n_freq=mel_spec.shape[0], hop_length=256)(mel_spec, speed_factor)

        # Energy modification
        mel_spec = mel_spec * energy_factor

        # Pitch shifting (this is a simple approximation, more advanced methods exist)
        if pitch_factor != 1.0:
            freq_bins = mel_spec.shape[0]
            shift = int(freq_bins * (pitch_factor - 1))
            mel_spec = torch.roll(mel_spec, shifts=shift, dims=0)
            if shift > 0:
                mel_spec[:shift, :] = mel_spec[shift, :]
            else:
                mel_spec[shift:, :] = mel_spec[shift-1, :]

        return mel_spec

    def save_audio(self, waveform: torch.Tensor, sample_rate: int, filename: str):
        waveform_numpy = waveform.cpu().numpy()
        scaled = np.int16(waveform_numpy / np.max(np.abs(waveform_numpy)) * 32767)
        write(filename, sample_rate, scaled)

def main():
    # Initialize the pipeline
    tts_pipeline = TextToSpeechPipeline(
        "microsoft/speecht5_tts",
        "microsoft/speecht5_hifigan"
    )

    # Example usage
    text = "Hello, how are you? This is an advanced text-to-speech system using Hugging Face models."

    waveform, sample_rate = tts_pipeline.generate_speech(
        text,
        speed_factor=1.1,  # Slightly faster
        energy_factor=1.2,  # Slightly louder
        pitch_factor=0.95  # Slightly lower pitch
    )

    # Save the generated audio
    tts_pipeline.save_audio(waveform, sample_rate, "output_speech.wav")

    print(f"Generated speech saved as 'output_speech.wav'")

    # Example of using a dataset from Hugging Face
    dataset = load_dataset("common_voice", "en", split="test", streaming=True)

    for sample in dataset.take(5):  # Process 5 samples
        text = sample["sentence"]
        print(f"Processing: {text}")
        waveform, sample_rate = tts_pipeline.generate_speech(text)
        tts_pipeline.save_audio(waveform, sample_rate, f"output_{sample['id']}.wav")

if __name__ == "__main__":
    main()
