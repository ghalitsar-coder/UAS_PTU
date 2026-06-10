#!/usr/bin/env python3
"""
Test script untuk memverifikasi perbaikan TTS di app.py
Mensimulasikan kedua pipeline tanpa Gradio UI
"""

import torch
import torchaudio
import librosa
import numpy as np
import re
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
from speechbrain.inference.speaker import EncoderClassifier
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# Load models
print("Loading models...")
processor_stt = Wav2Vec2Processor.from_pretrained("./stt-model-kustom-final")
model_stt = Wav2Vec2ForCTC.from_pretrained("./stt-model-kustom-final").to(DEVICE)

processor_tts = SpeechT5Processor.from_pretrained("./tts-model-kustom-final")
model_tts = SpeechT5ForTextToSpeech.from_pretrained("./tts-model-kustom-final").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": DEVICE},
    savedir="/tmp/speechbrain"
)

# Load speaker embedding
ref_audio = "voice-dataset/angka/angka1.wav"
audio, sr = torchaudio.load(ref_audio)
if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True)
if sr != 16000: audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
with torch.no_grad():
    default_embedding = spk_model.encode_batch(audio.to(DEVICE)).squeeze(0).squeeze(0)

print("Models loaded!\n")

# Middleware function
def process_math_text(text):
    replacements = {
        r'\bsatu\b': '1', r'\bdua\b': '2', r'\btiga\b': '3', r'\bempat\b': '4',
        r'\blima\b': '5', r'\benam\b': '6', r'\btujuh\b': '7', r'\bdelapan\b': '8',
        r'\bsembilan\b': '9', r'\bnol\b': '0',
        r'\btambah\b': '+', r'\bditambah\b': '+',
        r'\bkurang\b': '-', r'\bdikurangi\b': '-',
        r'\bkali\b': 'x', r'\bdikali\b': 'x',
        r'\bbagi\b': ':', r'\bdibagi\b': ':',
        r'\bsama dengan\b': '=', r'\bhasilnya\b': '='
    }
    processed_text = text.lower()
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text)
    processed_text = re.sub(r'\s+([+x=:-])\s+', r'\1', processed_text)
    return processed_text.strip()

# TTS function with fix
def generate_custom_tts(text, output_path):
    # Convert symbols back to words
    text_for_tts = text.replace('+', ' ditambah ').replace('=', ' sama dengan ')
    text_for_tts = text_for_tts.replace('x', ' dikali ').replace(':', ' dibagi ').replace('-', ' dikurangi ')
    
    num_map = {'1':'satu','2':'dua','3':'tiga','4':'empat','5':'lima','6':'enam','7':'tujuh','8':'delapan','9':'sembilan','0':'nol'}
    for n, word in num_map.items():
        text_for_tts = text_for_tts.replace(n, f" {word} ")
    
    text_for_tts = " ".join(text_for_tts.split())
    print(f"   TTS Input: '{text_for_tts}'")
    
    inputs = processor_tts(text=text_for_tts, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        speech = model_tts.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=default_embedding.unsqueeze(0),
            vocoder=vocoder,
        )
    
    # Normalize audio
    speech_np = speech.detach().cpu().numpy()
    max_val = np.abs(speech_np).max()
    print(f"   Max amplitude before norm: {max_val:.6f}")
    if max_val > 0:
        speech_np = speech_np / max_val
    print(f"   Max amplitude after norm: {np.abs(speech_np).max():.6f}")
    
    torchaudio.save(output_path, torch.tensor(speech_np).unsqueeze(0), 16000)
    print(f"   Saved: {output_path}\n")

# TEST 1: Math Pipeline
print("="*60)
print("TEST 1: MATH PIPELINE (Tab 1)")
print("="*60)
test_audio = "voice-dataset/angka/angka1.wav"  # Ganti dengan audio yang sesuai
audio_array, _ = librosa.load(test_audio, sr=16000)
input_values = processor_stt(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
with torch.no_grad():
    logits = model_stt(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor_stt.batch_decode(predicted_ids)[0]
math_text = process_math_text(transcription)

print(f"1. STT Output: '{transcription}'")
print(f"2. Math Format: '{math_text}'")
print(f"3. Generating TTS...")
generate_custom_tts(math_text, "test_pipeline_math.wav")

# TEST 2: Universal Pipeline (simulate with text)
print("="*60)
print("TEST 2: UNIVERSAL PIPELINE (Tab 2)")
print("="*60)
# Simulasi output Whisper
whisper_text = "halo dunia ini adalah tes suara"
print(f"1. Whisper Output: '{whisper_text}'")
print(f"2. Generating TTS...")
generate_custom_tts(whisper_text, "test_pipeline_universal.wav")

print("="*60)
print("✅ TEST SELESAI")
print("="*60)
print("File output:")
print("  - test_pipeline_math.wav")
print("  - test_pipeline_universal.wav")
print("\nSilakan dengarkan kedua file untuk verifikasi.")