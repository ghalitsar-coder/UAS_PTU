#!/usr/bin/env python3
"""
Debug script untuk memeriksa TTS di pipeline.
"""

import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier
import os
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# 1. Load TTS model
print("1. Loading TTS model...")
tts_kustom_path = "./tts-model-kustom-final"
if not os.path.exists(tts_kustom_path):
    print(f"ERROR: {tts_kustom_path} tidak ditemukan!")
    exit(1)

processor_tts = SpeechT5Processor.from_pretrained(tts_kustom_path)
model_tts = SpeechT5ForTextToSpeech.from_pretrained(tts_kustom_path).to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

# 2. Load speaker embedding
print("2. Loading speaker embedding...")
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": DEVICE},
    savedir=os.path.join("/tmp", "speechbrain")
)

ref_audio = "voice-dataset/angka/angka1.wav"
if not os.path.exists(ref_audio):
    print(f"ERROR: {ref_audio} tidak ditemukan!")
    exit(1)

audio, sr = torchaudio.load(ref_audio)
print(f"   Audio shape: {audio.shape}, Sample rate: {sr}")

if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)
    print(f"   Converted to mono: {audio.shape}")

if sr != 16000:
    print(f"   Resampling from {sr} to 16000...")
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

audio = audio.to(DEVICE)
print(f"   Audio on device: {audio.device}")

with torch.no_grad():
    speaker_embeddings = spk_model.encode_batch(audio.unsqueeze(0))
    print(f"   Speaker embedding shape before squeeze: {speaker_embeddings.shape}")
    speaker_embeddings = speaker_embeddings.squeeze(0).squeeze(0)
    print(f"   Speaker embedding shape after squeeze: {speaker_embeddings.shape}")
    print(f"   Speaker embedding dtype: {speaker_embeddings.dtype}")
    print(f"   Speaker embedding device: {speaker_embeddings.device}")

# 3. Test TTS dengan beberapa teks
test_texts = [
    "1+1=2",
    "satu dua tiga",
    "hello world",
    "1+1=2 1+2=3 3+1=4 123 sayang semuanya"
]

print("\n3. Testing TTS generation...")
for i, text in enumerate(test_texts):
    print(f"\n   Test {i+1}: '{text}'")
    
    inputs = processor_tts(text=text, return_tensors="pt")
    print(f"   Input IDs shape: {inputs['input_ids'].shape}")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        speech = model_tts.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings.unsqueeze(0),
            vocoder=vocoder,
        )
    
    print(f"   Speech shape: {speech.shape}")
    print(f"   Speech dtype: {speech.dtype}")
    print(f"   Speech device: {speech.device}")
    print(f"   Speech min: {speech.min().item():.6f}, max: {speech.max().item():.6f}")
    print(f"   Speech mean: {speech.mean().item():.6f}, std: {speech.std().item():.6f}")
    
    # Check if speech is all zeros or very small
    if torch.abs(speech).max() < 0.001:
        print(f"   WARNING: Speech values are very small (max abs: {torch.abs(speech).max().item():.6f})")
    
    # Save to file
    output_path = f"debug_tts_test_{i+1}.wav"
    speech_cpu = speech.detach().cpu().float()
    torchaudio.save(output_path, speech_cpu.unsqueeze(0), 16000)
    print(f"   Saved to: {output_path}")

print("\n4. Checking model configuration...")
print(f"   Model device: {model_tts.device}")
print(f"   Vocoder device: {vocoder.device}")
print(f"   Model config reduction_factor: {model_tts.config.reduction_factor}")

# 5. Test tanpa vocoder (hanya spectrogram)
print("\n5. Testing without vocoder (spectrogram only)...")
text = "test"
inputs = processor_tts(text=text, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    spectrogram = model_tts.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings.unsqueeze(0),
        vocoder=None,  # No vocoder
    )

print(f"   Spectrogram shape: {spectrogram.shape}")
print(f"   Spectrogram min: {spectrogram.min().item():.6f}, max: {spectrogram.max().item():.6f}")

# Save spectrogram
np.save("debug_spectrogram.npy", spectrogram.detach().cpu().numpy())
print(f"   Spectrogram saved to: debug_spectrogram.npy")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
print("Check the generated WAV files in the current directory.")
print("If files are silent, check speaker embedding extraction.")
print("If files have sound, check Gradio audio playback.")