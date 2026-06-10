import gradio as gr
import torch
import torchaudio
import librosa
import re
import numpy as np
import os
import warnings
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
from speechbrain.inference.speaker import EncoderClassifier
import whisper

warnings.filterwarnings('ignore')

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading models on {DEVICE}... Please wait.")

# ==========================================
# 1. LOAD STT KUSTOM (Math Focus)
# ==========================================
stt_kustom_path = "./stt-model-kustom-final"
if os.path.exists(stt_kustom_path):
    print("Loading Custom STT...")
    processor_stt = Wav2Vec2Processor.from_pretrained(stt_kustom_path)
    model_stt = Wav2Vec2ForCTC.from_pretrained(stt_kustom_path).to(DEVICE)
else:
    print(f"ERROR: {stt_kustom_path} tidak ditemukan!")
    model_stt, processor_stt = None, None

# ==========================================
# 2. LOAD STT UNIVERSAL (Whisper)
# ==========================================
print("Loading Universal STT (Whisper small)...")
model_whisper = whisper.load_model("small", device=DEVICE)

# ==========================================
# 3. LOAD TTS KUSTOM (SpeechT5)
# ==========================================
tts_kustom_path = "./tts-model-kustom-final"
if os.path.exists(tts_kustom_path):
    print("Loading Custom TTS & Vocoder...")
    processor_tts = SpeechT5Processor.from_pretrained(tts_kustom_path)
    model_tts = SpeechT5ForTextToSpeech.from_pretrained(tts_kustom_path).to(DEVICE)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
    
    print("Loading SpeechBrain Speaker Extractor...")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": DEVICE},
        savedir=os.path.join("/tmp", "speechbrain")
    )
else:
    print(f"ERROR: {tts_kustom_path} tidak ditemukan!")
    processor_tts, model_tts, vocoder, spk_model = None, None, None, None

# Load Default Speaker Embedding dari dataset
default_embedding = None
# Use a longer audio file for better speaker embedding (full sentence)
ref_audio = "voice-dataset/kalimatmtk/kalimatmtk1.wav"
if not os.path.exists(ref_audio):
    # Fallback to the short audio if the long one doesn't exist
    ref_audio = "voice-dataset/angka/angka1.wav"
if spk_model and os.path.exists(ref_audio):
    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    with torch.no_grad():
        default_embedding = spk_model.encode_batch(audio.to(DEVICE)).squeeze(0).squeeze(0)
    print(f"Loaded speaker embedding from: {ref_audio}")

# ==========================================
# LOGIC / MIDDLEWARE
# ==========================================
def process_math_text(text):
    """Mengubah kata menjadi simbol matematika."""
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
    
    # Menghapus spasi berlebih di sekitar simbol
    processed_text = re.sub(r'\s+([+x=:-])\s+', r'\1', processed_text)
    return processed_text.strip()

# ==========================================
# PIPELINES
# ==========================================
def generate_custom_tts(text):
    """Generate audio from text using custom SpeechT5."""
    if not model_tts or not text: return None
    
    # SPEECHT5 FIX: Ubah angka dan simbol kembali ke kata-kata agar bisa disuarakan
    # Tokenizer SpeechT5 tidak mengenali simbol '+', '=', dll.
    text_for_tts = text.replace('+', ' ditambah ').replace('=', ' sama dengan ')
    text_for_tts = text_for_tts.replace('x', ' dikali ').replace(':', ' dibagi ').replace('-', ' dikurangi ')
    
    # Mapping angka ke kata (sederhana)
    num_map = {'1':'satu','2':'dua','3':'tiga','4':'empat','5':'lima','6':'enam','7':'tujuh','8':'delapan','9':'sembilan','0':'nol'}
    for n, word in num_map.items():
        text_for_tts = text_for_tts.replace(n, f" {word} ")
    
    # Bersihkan spasi ganda
    text_for_tts = " ".join(text_for_tts.split())
    
    print(f"TTS Input: '{text_for_tts}'")
    
    inputs = processor_tts(text=text_for_tts, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        speech = model_tts.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=default_embedding.unsqueeze(0),
            vocoder=vocoder,
        )
    
    # NORMALISASI + TRIM SILENCE AWAL
    speech_np = speech.detach().cpu().numpy()
    
    # Trim silence di awal (threshold 0.01 RMS per frame 20ms)
    frame_size = int(16000 * 0.02)  # 20ms frames
    trimmed_start = 0
    for i in range(0, len(speech_np) - frame_size, frame_size):
        frame = speech_np[i:i + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > 0.005:
            # Sedikit mundur agar tidak memotong terlalu agresif
            trimmed_start = max(0, i - frame_size)
            break
    speech_np = speech_np[trimmed_start:]
    
    # Normalisasi ke [-1, 1]
    max_val = np.abs(speech_np).max()
    if max_val > 0:
        speech_np = speech_np / max_val
    
    return (16000, speech_np)

def run_math_pipeline(audio_path):
    """Pipeline Tab 1: Kustom STT -> Math Middleware -> Kustom TTS"""
    if audio_path is None: return "Silakan masukkan audio.", "Silakan masukkan audio.", None
    if not model_stt: return "Model STT Kustom Error", "", None
    
    # 1. STT
    audio_array, _ = librosa.load(audio_path, sr=16000)
    input_values = processor_stt(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model_stt(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_stt.batch_decode(predicted_ids)[0]
    
    # 2. Middleware
    math_text = process_math_text(transcription)
    
    # 3. TTS
    audio_out = generate_custom_tts(math_text)
    
    return transcription, math_text, audio_out

def run_universal_pipeline(audio_path):
    """Pipeline Tab 2: Whisper STT -> Raw Text -> Kustom TTS"""
    if audio_path is None: return "Silakan masukkan audio.", None
    
    # 1. STT Whisper
    result = model_whisper.transcribe(audio_path, language="id")
    transcription = result["text"].strip()
    
    # 2. TTS Kustom
    audio_out = generate_custom_tts(transcription)
    
    return transcription, audio_out

# ==========================================
# GRADIO UI
# ==========================================
with gr.Blocks(title="STT & TTS Terintegrasi") as app:
    gr.Markdown("# 🤖 Sistem STT & TTS Terintegrasi")
    gr.Markdown("**Proyek Tugas Akhir Pengantar Teknologi Informasi (PTU)**")
    
    with gr.Tabs():
        # TAB 1
        with gr.TabItem("🧮 Mode Tugas Akhir (Math Focus)"):
            gr.Markdown("Pipeline: Mic -> STT Kustom (Wav2Vec2) -> Math Middleware -> TTS Kustom (SpeechT5) -> Speaker")
            
            with gr.Row():
                with gr.Column():
                    audio_in_math = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                    btn_math = gr.Button("🚀 Proses Audio", variant="primary")
                
                with gr.Column():
                    text_raw_math = gr.Textbox(label="1. Transkripsi Mentah (STT Kustom)")
                    text_final_math = gr.Textbox(label="2. Hasil Middleware (Math Format)")
                    audio_out_math = gr.Audio(label="3. Output TTS (Voice Cloning)", interactive=False)
            
            btn_math.click(
                fn=run_math_pipeline,
                inputs=audio_in_math,
                outputs=[text_raw_math, text_final_math, audio_out_math]
            )
            
        # TAB 2
        with gr.TabItem("🎙️ Mode Voice Cloning Bebas (Playground)"):
            gr.Markdown("Pipeline: Mic -> STT Universal (Whisper) -> TTS Kustom (SpeechT5) -> Speaker")
            
            with gr.Row():
                with gr.Column():
                    audio_in_univ = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio Bebas")
                    btn_univ = gr.Button("🚀 Proses Audio", variant="primary")
                
                with gr.Column():
                    text_univ = gr.Textbox(label="1. Transkripsi Universal (Whisper)")
                    audio_out_univ = gr.Audio(label="2. Output TTS (Voice Cloning)", interactive=False)
                    
            btn_univ.click(
                fn=run_universal_pipeline,
                inputs=audio_in_univ,
                outputs=[text_univ, audio_out_univ]
            )

if __name__ == "__main__":
    print("Starting Gradio server...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)