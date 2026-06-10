# Panduan Integrasi STT + TTS End-to-End

## Ringkasan
Dokumen ini menjelaskan cara mengintegrasikan model Speech-to-Text (STT) dan Text-to-Speech (TTS) kustom menjadi satu pipeline yang dapat memproses audio input → teks → audio output dengan suara kustom.

## Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT AUDIO (WAV)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  1. STT (Wav2Vec2)             │
        │  Input: Audio (16kHz, mono)    │
        │  Output: Teks transkripsi      │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  2. TEXT PROCESSING            │
        │  Input: Teks mentah            │
        │  Output: Teks terformat        │
        │  (angka, simbol matematika)    │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  3. TTS (SpeechT5 + HiFi-GAN)  │
        │  Input: Teks terformat         │
        │  Output: Audio (16kHz, mono)   │
        └────────────┬───────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT AUDIO (WAV)                        │
└─────────────────────────────────────────────────────────────┘
```

## Komponen yang Sudah Siap

### 1. STT Model (Wav2Vec2)
- **Model:** `indonesian-nlp/wav2vec2-large-xlsr-indonesian`
- **Status:** ✅ Sudah di-fine-tune
- **File:** `stt-model-kustom-final/`
- **Input:** Audio WAV (16kHz, mono)
- **Output:** Teks transkripsi

### 2. Text Processing (Middleware)
- **Status:** ✅ Sudah ada di `convert.py`
- **Fungsi:** Konversi teks mentah → teks terformat
- **Contoh:**
  - "satu tambah satu sama dengan dua" → "1+1=2"
  - "seratus dua puluh tiga" → "123"

### 3. TTS Model (SpeechT5)
- **Model:** `microsoft/speecht5_tts` (fine-tuned)
- **Status:** ✅ Sudah di-fine-tune dengan suara kustom
- **File:** `tts-model-kustom-final/`
- **Vocoder:** `microsoft/speecht5_hifigan`
- **Input:** Teks
- **Output:** Audio WAV (16kHz, mono)

## File yang Akan Dibuat

### `pipeline_end_to_end.py` (MAIN INTEGRATION)
Script utama yang menggabungkan STT → Text Processing → TTS

**Fitur:**
- Load semua model (STT, TTS, Vocoder)
- Process audio input end-to-end
- Output audio dengan suara kustom
- Error handling dan logging

### `app_gradio.py` (UI INTERFACE)
Interface web interaktif menggunakan Gradio

**Fitur:**
- Upload audio atau record langsung
- Real-time processing
- Display teks transkripsi
- Play output audio
- Download hasil

## Langkah Implementasi

### Step 1: Buat `pipeline_end_to_end.py`
Script yang mengintegrasikan semua komponen

### Step 2: Buat `app_gradio.py`
Interface web untuk demo

### Step 3: Testing
Uji dengan berbagai input audio

### Step 4: Dokumentasi
Buat panduan penggunaan untuk user

## Dependencies yang Diperlukan

Semua sudah ada di `requirements.txt`:
- `transformers` - Model loading
- `torch`, `torchaudio` - Audio processing
- `datasets` - Data handling
- `speechbrain` - Speaker embeddings
- `gradio` - Web UI
- `soundfile` - Audio I/O

## Next Steps

1. ✅ Fase 3 selesai (TTS + Vocoder)
2. ⏳ Fase 4a: Buat `pipeline_end_to_end.py`
3. ⏳ Fase 4b: Buat `app_gradio.py`
4. ⏳ Fase 4c: Testing & Documentation
