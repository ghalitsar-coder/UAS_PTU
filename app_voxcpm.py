import gradio as gr
import torch
import torchaudio
import librosa
import re
import numpy as np
import os
import warnings
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from voxcpm import VoxCPM
from speechbrain.inference.speaker import EncoderClassifier

# Config
# STT models (Wav2Vec2 + Whisper) dijalankan di CPU agar VRAM bebas untuk VoxCPM2
DEVICE = "cpu"
# VoxCPM2 mendapat GPU penuh untuk inferensi cepat (~3-5 detik vs ~30-60 detik di CPU)
VOXCPM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_STT_KUSTOM = "./stt-model-kustom-final"
MODEL_VOXCPM = "./pretrained_models/VoxCPM2"
REF_AUDIO = "voice-dataset/kalimatmtk/kalimatmtk1.wav"

print(f"Loading models on {DEVICE}... Please wait.")

# 1. Load Custom STT (Wav2Vec2)
print("Loading Custom STT...")
processor_stt = Wav2Vec2Processor.from_pretrained(MODEL_STT_KUSTOM)
model_stt = Wav2Vec2ForCTC.from_pretrained(MODEL_STT_KUSTOM).to(DEVICE)

# 2. Load Universal STT (Whisper)
print("Loading Universal STT (Whisper small)...")
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=DEVICE)

# 3. Load VoxCPM2
print("Loading VoxCPM2 on GPU (for maximum speed)...")
# Kita load model hanya jika file model.safetensors sudah ada
if os.path.exists(os.path.join(MODEL_VOXCPM, "model.safetensors")):
    model_tts = VoxCPM.from_pretrained(MODEL_VOXCPM, load_denoiser=False, device=VOXCPM_DEVICE)
else:
    model_tts = None
    print("WARNING: VoxCPM2 model file not found. TTS will be disabled.")

# ==========================================
# LOGIC / MIDDLEWARE
# ==========================================

def process_math_text(text):
    """Mengubah teks transkripsi menjadi format matematika simbolik."""
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
    
    # Hapus spasi di sekitar operator
    processed_text = re.sub(r'\s+([+x=:-])\s+', r'\1', processed_text)
    return processed_text.strip()

def num_to_id_words(n):
    """Convert integer 0-999 into Indonesian words for TTS pronunciation."""
    words = {
        0: "nol", 1: "satu", 2: "dua", 3: "tiga", 4: "empat", 5: "lima",
        6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan", 10: "sepuluh",
        11: "sebelas", 12: "dua belas", 13: "tiga belas", 14: "empat belas",
        15: "lima belas", 16: "enam belas", 17: "tujuh belas", 18: "delapan belas",
        19: "sembilan belas",
    }
    if n < 0:
        return "minus " + num_to_id_words(-n)
    if n < 20:
        return words[n]
    if n < 100:
        tens = n // 10
        ones = n % 10
        tens_word = ["", "", "dua puluh", "tiga puluh", "empat puluh", "lima puluh",
                     "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"][tens]
        return tens_word if ones == 0 else f"{tens_word} {words[ones]}"
    if n == 100:
        return "seratus"
    if n < 200:
        return "seratus " + num_to_id_words(n - 100)
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        hundreds_word = f"{words[hundreds]} ratus"
        return hundreds_word if rest == 0 else f"{hundreds_word} {num_to_id_words(rest)}"
    # fallback for large numbers: digit by digit
    return " ".join(num_to_id_words(int(d)) for d in str(n))


def convert_numbers_to_id_words(text):
    """Convert all numeric sequences in text into Indonesian words."""
    def repl(match):
        return num_to_id_words(int(match.group(0)))
    return re.sub(r"\d+", repl, text)


def generate_voxcpm_tts(text):
    """Generate audio from text using VoxCPM2 with Voice Cloning."""
    if model_tts is None:
        return None
    
    # Mapper simbol->kata untuk keamanan kualitas pengucapan matematis
    text_for_tts = text.replace('+', ' ditambah ').replace('=', ' sama dengan ')
    text_for_tts = text_for_tts.replace('x', ' dikali ').replace(':', ' dibagi ').replace('-', ' dikurangi ')
    
    # Gunakan fungsi yang lebih cerdas untuk mengubah angka berapapun menjadi kata
    text_for_tts = convert_numbers_to_id_words(text_for_tts)
    text_for_tts = " ".join(text_for_tts.split())
    
    print(f"TTS Input: '{text_for_tts}'")
    
    # Generate audio
    wav = model_tts.generate(
        text=text_for_tts,
        reference_wav_path=REF_AUDIO,
        cfg_value=2.0,
        inference_timesteps=10,
    )
    
    return (model_tts.tts_model.sample_rate, wav)

# ==========================================
# GRADIO HANDLERS
# ==========================================

def handle_math_mode(audio_path):
    if audio_path is None: return "No audio detected", None
    
    # 1. STT Kustom
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor_stt(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model_stt(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_stt.batch_decode(predicted_ids)[0]
    
    # 2. Middleware
    math_result = process_math_text(transcription)
    
    # 3. TTS Kustom (VoxCPM2)
    tts_output = generate_voxcpm_tts(math_result)
    
    return math_result, tts_output

def handle_playground_mode(audio_path):
    if audio_path is None: return "No audio detected", None
    
    # 1. STT Universal (Whisper)
    result = whisper_pipeline(audio_path)
    transcription = result["text"]
    
    # 2. TTS Kustom (VoxCPM2)
    tts_output = generate_voxcpm_tts(transcription)
    
    return transcription, tts_output

# ==========================================
# UI INTERFACE
# ==========================================

with gr.Blocks(title="UAS PTU: Integrated STT-TTS with VoxCPM2") as demo:
    gr.Markdown("# 🗣️ UAS PTU: Sistem STT & TTS Terintegrasi")
    gr.Markdown("Lead Engineer: Galitsar Gyasi Elfaris (ITENAS)")
    
    with gr.Tabs():
        # TAB 1: Math Focus
        with gr.Tab("Mode Tugas Akhir (Math Focus)"):
            gr.Markdown("### 🧮 Pipeline: Mic / Upload -> STT Kustom -> Math Middleware -> VoxCPM2 TTS")
            audio_input_1 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Suara (Record atau Upload .wav)")
            text_output_1 = gr.Textbox(label="Hasil Transkripsi (Matematika Simbolik)")
            audio_output_1 = gr.Audio(label="Output TTS (Voice Cloning)")
            btn_1 = gr.Button("Proses Audio", variant="primary")
            btn_1.click(handle_math_mode, inputs=audio_input_1, outputs=[text_output_1, audio_output_1])
            gr.Examples(
                examples=[["voice-dataset/kalimatmtk/kalimatmtk1.wav"]],
                inputs=audio_input_1,
                label="Contoh Audio (klik untuk load)"
            )

        # TAB 2: Playground
        with gr.Tab("Mode Voice Cloning Bebas (Playground)"):
            gr.Markdown("### 🎭 Pipeline: Mic / Upload -> STT Universal (Whisper) -> VoxCPM2 TTS")
            audio_input_2 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Suara (Record atau Upload .wav)")
            text_output_2 = gr.Textbox(label="Hasil Transkripsi (Teks Asli)")
            audio_output_2 = gr.Audio(label="Output TTS (Voice Cloning)")
            btn_2 = gr.Button("Proses Audio", variant="primary")
            btn_2.click(handle_playground_mode, inputs=audio_input_2, outputs=[text_output_2, audio_output_2])
            gr.Examples(
                examples=[["voice-dataset/kalimatmtk/kalimatmtk2.wav"]],
                inputs=audio_input_2,
                label="Contoh Audio (klik untuk load)"
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
