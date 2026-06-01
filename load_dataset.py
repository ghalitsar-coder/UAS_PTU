from datasets import load_dataset
import librosa

def test_load_dataset():
    # Memuat dataset dari file CSV
    print("Memuat dataset...")
    dataset = load_dataset('csv', data_files='metadata.csv', split='train')
    
    print(f"Berhasil memuat {len(dataset)} baris data!")
    
    # Mengambil sampel pertama untuk diuji
    sample = dataset[0]
    audio_path = sample['file_name']
    transcription = sample['transcription']
    
    print("\n--- Sampel Data Pertama ---")
    print(f"Path Audio   : {audio_path}")
    print(f"Transkripsi  : {transcription}")
    
    # Menguji pembacaan file audio dengan librosa pada 16kHz
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        print(f"Status Audio : Berhasil dibaca! (Sample rate: {sampling_rate}Hz)")
        print(f"Bentuk Array : {audio_array.shape}")
    except Exception as e:
        print(f"Gagal membaca audio: {e}")

if __name__ == "__main__":
    test_load_dataset()