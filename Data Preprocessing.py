import torchaudio.transforms as T

# a  funcation to porcccses text
def preprocess_text(text):
    return text.lower()
  
# funcation to proccses audio
def preprocess_audio(audio):
    waveform, sample_rate = audio
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=80)
    mel_spec = transform(waveform)
    return mel_spec

dataset = dataset.map(lambda x: {"text": preprocess_text(x["text"]),
                                 "mel_spectrogram": preprocess_audio((x["audio"]["array"], x["audio"]["sampling_rate"]))},
                      remove_columns=["audio"])