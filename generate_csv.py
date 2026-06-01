import os
import glob
import csv

def generate_metadata_template():
    dataset_dir = "voice-dataset"
    output_csv = "metadata.csv"
    
    # Mencari semua file .wav di dalam subfolder voice-dataset
    # Menggunakan recursive=True agar membaca folder angka, fonetik, dll
    search_pattern = os.path.join(dataset_dir, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    if not wav_files:
        print("Tidak ada file .wav yang ditemukan.")
        return

    # Menulis ke file CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "transcription"]) # Header
        
        for wav_path in wav_files:
            # Mengubah backslash (Windows) menjadi forward slash agar aman saat dilatih
            normalized_path = wav_path.replace("\\", "/")
            # Mengisi kolom transkripsi dengan string kosong sebagai template
            writer.writerow([normalized_path, ""])
            
    print(f"Berhasil membuat {output_csv} dengan {len(wav_files)} baris audio.")
    print("Silakan buka file tersebut dan isi kolom 'transcription'.")

if __name__ == "__main__":
    generate_metadata_template()