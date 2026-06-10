#!/usr/bin/env python3
"""
Inference script untuk model SpeechT5 kustom hasil fine-tuning.
Menggunakan model yang sudah dilatih di `tts-model-kustom-final/`.
"""

import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_processor():
    """Memuat model dan processor dari checkpoint lokal."""
    print("1. Memuat Processor dan Model SpeechT5 kustom...")
    processor = SpeechT5Processor.from_pretrained("./tts-model-kustom-final")
    model = SpeechT5ForTextToSpeech.from_pretrained("./tts-model-kustom-final")
    
    print("2. Memuat Vocoder HiFi-GAN resmi SpeechT5...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    print("3. Memuat Ekstraktor Karakteristik Suara (SpeechBrain)...")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        savedir=os.path.join("/tmp", "speechbrain")
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    vocoder = vocoder.to(device)
    model.eval()
    vocoder.eval()
    
    return processor, model, vocoder, spk_model

def extract_speaker_embedding(audio_path, spk_model):
    """Mengekstrak speaker embedding dari audio referensi."""
    print(f"3. Mengekstrak speaker embedding dari {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    
    # Preprocessing audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze()
    
    # Ekstrak embedding
    with torch.no_grad():
        speaker_embeddings = spk_model.encode_batch(audio.unsqueeze(0))
        speaker_embeddings = speaker_embeddings.squeeze(0).squeeze(0)
    
    print(f"   Speaker embedding shape: {speaker_embeddings.shape}")
    return speaker_embeddings

def synthesize_text(text, processor, model, vocoder, speaker_embeddings, output_path="hasil_tts_speecht5.wav"):
    """Mensintesis teks langsung menjadi audio WAV menggunakan SpeechT5 + HiFi-GAN."""
    print(f"4. Mensintesis teks: '{text}'...")
    
    inputs = processor(text=text, return_tensors="pt")
    
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    speaker_embeddings = speaker_embeddings.to(device)
    
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings.unsqueeze(0),
            vocoder=vocoder,
        )
    
    speech = speech.detach().cpu().float()
    torchaudio.save(output_path, speech.unsqueeze(0), 16000)
    
    print(f"   Audio tensor shape: {speech.shape}")
    print(f"   Durasi audio: {speech.numel() / 16000:.2f} detik")
    print(f"   Audio disimpan sebagai: {output_path}")
    return output_path

def main():
    # Teks target dari PRD
    target_text = "1+1=2 1+2=3 3+1=4 123 sayang semuanya"
    
    # Load model
    processor, model, vocoder, spk_model = load_model_and_processor()
    
    # Gunakan audio referensi dari dataset (misal: suara angka 1)
    reference_audio = "voice-dataset/angka/angka1.wav"
    if not os.path.exists(reference_audio):
        print(f"Warning: {reference_audio} tidak ditemukan.")
        print("Menggunakan audio pertama dari dataset...")
        import glob
        audio_files = glob.glob("voice-dataset/**/*.wav", recursive=True)
        if audio_files:
            reference_audio = audio_files[0]
        else:
            print("Error: Tidak ada file audio referensi ditemukan.")
            return
    
    # Ekstrak speaker embedding
    speaker_embeddings = extract_speaker_embedding(reference_audio, spk_model)
    
    # Synthesize text langsung ke audio
    output_file = synthesize_text(target_text, processor, model, vocoder, speaker_embeddings)
    
    print("\n" + "="*60)
    print("✅ INFERENCE SPEECHT5 + HIFIGAN SELESAI!")
    print("="*60)
    print(f"Model TTS: tts-model-kustom-final/")
    print(f"Vocoder: microsoft/speecht5_hifigan")
    print(f"Teks: {target_text}")
    print(f"Audio referensi: {reference_audio}")
    print(f"Output audio: {output_file}")
    print("\nLangkah selanjutnya:")
    print("1. Integrasi dengan pipeline STT untuk sistem end-to-end")
    print("2. Bangun UI dengan Gradio/Streamlit")

if __name__ == "__main__":
    main()