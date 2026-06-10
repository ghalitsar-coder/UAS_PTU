from supertonic import TTS

def test_tts():
    print("1. Memuat model Supertonic-3 TTS (dan mengunduh jika belum ada)...")
    tts = TTS(auto_download=True)
    
    print("2. Menyiapkan Voice Style...")
    # Menggunakan preset suara bawaan (M1 = Male 1)
    style = tts.get_voice_style(voice_name="M1")
    
    # Masukkan Teks Mentah dari output STT kamu sebelumnya
    teks_masukan = "satu tambah satu sama dengan dua, satu dua tiga sayang semuanya"
    
    print(f"3. Mensintesis suara untuk teks: '{teks_masukan}'...")
    # Pastikan menggunakan lang="id" untuk Bahasa Indonesia
    wav, duration = tts.synthesize(teks_masukan, voice_style=style, lang="id")
    
    output_file = "hasil_tts.wav"
    tts.save_audio(wav, output_file)
    print(f"\n✅ Selesai! Audio tersimpan sebagai {output_file}")
if __name__ == "__main__":
    test_tts()