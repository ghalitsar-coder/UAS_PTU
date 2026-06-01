import torch
import librosa
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def process_text(text):
    # Dictionary mapping untuk mengubah kata menjadi simbol matematika
    replacements = {
        r'\bsatu\b': '1',
        r'\bdua\b': '2',
        r'\btiga\b': '3',
        r'\bempat\b': '4',
        r'\blima\b': '5',
        r'\benam\b': '6',
        r'\btujuh\b': '7',
        r'\bdelapan\b': '8',
        r'\bsembilan\b': '9',
        r'\bnol\b': '0',
        r'\btambah\b': '+',
        r'\bditambah\b': '+',
        r'\bkurang\b': '-',
        r'\bdikurangi\b': '-',
        r'\bkali\b': 'x',
        r'\bdikali\b': 'x',
        r'\bbagi\b': ':',
        r'\bdibagi\b': ':',
        r'\bsama dengan\b': '=',
        r'\bhasilnya\b': '='
    }
    
    # Menerapkan regex replacement
    processed_text = text
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text)
    
    # Menghapus spasi berlebih di sekitar simbol matematika
    processed_text = re.sub(r'\s+([+x=:])\s+', r'\1', processed_text)
    processed_text = re.sub(r'([+x=:])\s+', r'\1', processed_text)
    processed_text = re.sub(r'\s+([+x=:])', r'\1', processed_text)
    
    return processed_text

def main():
    print("Memuat model STT kustom...")
    model_path = "./stt-model-kustom-final"
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Pilih salah satu file audio untuk diuji
    # Pastikan path ini sesuai dengan nama file di dalam folder datasetmu
    test_audio = "voice-dataset/kalimat/kalimat1.wav" 
    
    print(f"Membaca file audio: {test_audio}")
    try:
        audio_array, _ = librosa.load(test_audio, sr=16000)
    except Exception as e:
        print(f"Gagal memuat audio: {e}")
        return

    # Prediksi STT
    input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    print("\n--- HASIL INFERENSI ---")
    print(f"Teks Mentah (STT) : {transcription}")
    
    # Memasukkan ke Middleware
    final_output = process_text(transcription)
    print(f"Teks Final (Math) : {final_output}")

if __name__ == "__main__":
    main()