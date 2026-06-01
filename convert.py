import os
import subprocess
import glob

def convert_m4a_to_wav():
    # Menghapus '/' di awal agar menjadi relative path
    m4a_files = glob.glob("voice-dataset/*.m4a")
    
    if not m4a_files:
        print("Tidak ditemukan file .m4a di direktori ini.")
        return

    print(f"Ditemukan {len(m4a_files)} file. Memulai konversi...")

    for file in m4a_files:
        # Membuat nama file output (.wav)
        filename_without_ext = os.path.splitext(file)[0]
        output_file = f"{filename_without_ext}.wav"
        
        # Perintah FFmpeg: -ac 1 (Mono), -ar 16000 (16kHz sample rate)
        command = [
            "ffmpeg", 
            "-i", file, 
            "-ac", "1", 
            "-ar", "16000", 
            output_file
        ]
        
        try:
            # Menjalankan FFmpeg tanpa menampilkan log yang terlalu panjang
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"Berhasil: {file} -> {output_file}")
        except subprocess.CalledProcessError:
            print(f"Gagal mengonversi: {file}")

if __name__ == "__main__":
    convert_m4a_to_wav()