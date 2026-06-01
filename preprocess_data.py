from datasets import load_dataset
from transformers import Wav2Vec2Processor
import librosa
import warnings

# Mengabaikan warning deprecation agar terminal tetap bersih
warnings.filterwarnings('ignore')

def main():
    print("1. Memuat dataset...")
    dataset = load_dataset('csv', data_files='metadata.csv', split='train')
    
    print("2. Mengunduh dan memuat Processor Wav2Vec2...")
    # Menggunakan model bahasa Indonesia yang direkomendasikan
    processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian")

    def prepare_dataset(batch):
        # Membaca file audio
        audio_array, _ = librosa.load(batch["file_name"], sr=16000)
        
        # Mengubah audio menjadi input_values (array numerik yang dipahami model)
        batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
        
        # Mengubah teks transkripsi menjadi ID label (token)
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        
        return batch

    print("3. Memetakan dataset (ekstraksi fitur audio & tokenisasi teks)...")
    # map() akan menjalankan fungsi prepare_dataset ke setiap baris secara otomatis
    mapped_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    print("\n✅ Selesai! Dataset telah diubah menjadi tensor siap latih.")
    print(f"Struktur data baru: {mapped_dataset.column_names}")

if __name__ == "__main__":
    main()