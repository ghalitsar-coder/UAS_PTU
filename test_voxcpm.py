#!/usr/bin/env python3
"""
Test VoxCPM2 untuk voice cloning suara kustom.
Menggunakan kalimatmtk1.wav sebagai referensi suara.
"""

from voxcpm import VoxCPM
import soundfile as sf
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"VRAM tersedia: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB" if torch.cuda.is_available() else "CPU mode")

MODEL_PATH = "./pretrained_models/VoxCPM2"
REF_AUDIO   = "voice-dataset/kalimatmtk/kalimatmtk1.wav"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model tidak ditemukan di {MODEL_PATH}")
    exit(1)

if not os.path.exists(REF_AUDIO):
    print(f"ERROR: Audio referensi tidak ditemukan: {REF_AUDIO}")
    exit(1)

print(f"\nMemuat VoxCPM2 dari {MODEL_PATH}...")
model = VoxCPM.from_pretrained(
    MODEL_PATH,
    load_denoiser=False,   # hemat VRAM
    device=DEVICE,
)
print("Model loaded!\n")

# ---- Test 1: Controllable Voice Cloning ----
print("=" * 60)
print("TEST 1: Voice Cloning - Kalimat Math")
print("=" * 60)
text1 = "satu ditambah satu sama dengan dua. satu ditambah dua sama dengan tiga."
print(f"Teks: {text1}")
wav1 = model.generate(
    text=text1,
    reference_wav_path=REF_AUDIO,
    cfg_value=2.0,
    inference_timesteps=10,
)
out1 = "voxcpm_test_math.wav"
sf.write(out1, wav1, model.tts_model.sample_rate)
print(f"Saved: {out1} ({model.tts_model.sample_rate}Hz)\n")

# ---- Test 2: Voice Cloning - Kalimat Bebas ----
print("=" * 60)
print("TEST 2: Voice Cloning - Kalimat Bebas")
print("=" * 60)
text2 = "Halo, nama saya Galitsar. Ini adalah demonstrasi sistem text to speech dengan voice cloning."
print(f"Teks: {text2}")
wav2 = model.generate(
    text=text2,
    reference_wav_path=REF_AUDIO,
    cfg_value=2.0,
    inference_timesteps=10,
)
out2 = "voxcpm_test_bebas.wav"
sf.write(out2, wav2, model.tts_model.sample_rate)
print(f"Saved: {out2} ({model.tts_model.sample_rate}Hz)\n")

# ---- Test 3: Ultimate Cloning (dengan transcript referensi) ----
print("=" * 60)
print("TEST 3: Ultimate Cloning (audio + transcript)")
print("=" * 60)
# Ganti REF_TEXT dengan isi transkripsi kalimatmtk1.wav kamu
REF_TEXT = "dua ditambah dua sama dengan empat"  # transkripsi dari kalimatmtk1.wav
text3 = "tiga tambah empat sama dengan tujuh. dua kali lima sama dengan sepuluh."
print(f"Ref text: {REF_TEXT}")
print(f"Teks: {text3}")
wav3 = model.generate(
    text=text3,
    prompt_wav_path=REF_AUDIO,
    prompt_text=REF_TEXT,
    reference_wav_path=REF_AUDIO,
)
out3 = "voxcpm_test_ultimate.wav"
sf.write(out3, wav3, model.tts_model.sample_rate)
print(f"Saved: {out3} ({model.tts_model.sample_rate}Hz)\n")

print("=" * 60)
print("✅ SEMUA TEST VOXCPM2 SELESAI")
print("=" * 60)
print(f"Output files:")
for f in [out1, out2, out3]:
    size = os.path.getsize(f) / 1024
    print(f"  {f} ({size:.0f}KB)")
