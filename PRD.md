# Product Requirements Document (PRD)
**Proyek:** Sistem Kustom Speech-to-Text (STT) & Text-to-Speech (TTS) Terintegrasi
**Mata Kuliah:** Tugas Akhir Pengantar Teknologi Informasi (PTU) - Proyek Kelompok
**Lead Engineer / Developer:** Galitsar Gyasi Elfaris
**Institusi:** Institut Teknologi Nasional (ITENAS), Bandung

## 1. Ringkasan Eksekutif
Proyek ini bertujuan untuk membangun sebuah sistem kecerdasan buatan (*Artificial Intelligence*) *end-to-end* yang mampu memproses input suara pengguna, mentranskripsikannya menjadi teks (termasuk konversi simbol matematika), dan menyuarakannya kembali menggunakan model *voice cloning* kustom. Sistem dirancang dengan arsitektur hibrida untuk mengakomodasi kebutuhan tugas spesifik maupun fungsionalitas universal.

## 2. Tujuan dan Target Keluaran
* **Akurasi Transkripsi Spesifik:** Model STT kustom dapat mengenali operasi matematika bahasa Indonesia.
* **Transkripsi Universal:** Model STT sekunder dapat menangkap percakapan umum.
* **Pemrosesan Konteks:** Sistem harus mampu merapikan teks (misal: "tambah" menjadi `+`, "sama dengan" menjadi `=`).
* **Kloning Suara:** Model TTS harus bisa mensintesis teks menggunakan *Voice Cloning* asli suara pengembang.

## 3. Arsitektur Sistem & Teknologi
* **Input/Output:** Mikrofon Web / File Audio `.wav` (Mono, 16kHz).
* **Model Telinga (STT) - Dual Mode:**
  1. *Custom Module:* `Wav2Vec2-large-xlsr-indonesian` (Di-fine-tune dengan dataset lokal matematika).
  2. *Universal Module:* `openai/whisper-small` atau `base` (Pre-trained universal untuk percakapan bebas).
* **Middleware (Logika Python):** Algoritma Regex / Dictionary Mapping untuk *parsing* simbol matematis.
* **Model Mulut (TTS):** `Microsoft/SpeechT5_TTS` (Arsitektur Voice Cloning dengan ekstraksi Speaker Embedding).
* **Render Audio (Vocoder):** `SpeechT5HifiGan` (Konversi Mel-Spectrogram ke Waveform).
* **Antarmuka (UI):** Gradio (berbasis Python) dengan arsitektur *Multi-Tab*.
* **Rencana Infrastruktur (DevOps):** Kontainerisasi layanan menggunakan Docker, target orkestrasi Kubernetes di *cloud* (Azure/AWS).

## 4. Fase Implementasi dan Status (Checklist)

### Fase 1: Persiapan Dataset (Data Collection) ✅
- [x] Merekam suara huruf, angka, frasa matematika, dan kalimat fonetik umum.
- [x] Konversi seluruh format audio ke `.wav` (Mono, 16kHz) menggunakan FFmpeg.
- [x] Menyusun *metadata.csv*.

### Fase 2: Pengembangan STT Spesifik (Wav2Vec2) ✅
- [x] *Preprocessing* dataset audio dan tokenisasi teks.
- [x] Mengunduh *weights* dan menyesuaikan *layer* klasifikasi untuk 30 token unik.
- [x] Proses *Fine-Tuning* dengan `Trainer` API (Loss: 1.199).
- [x] Uji coba Inferensi (STT berhasil mentranskripsi teks).

### Fase 3: Pengembangan TTS Kustom (Voice Cloning dengan SpeechT5) ✅
- [x] Instalasi *library* ekstraktor karakteristik suara (`speechbrain`).
- [x] Mengekstrak 512-dim *Speaker Embedding* (X-Vector) dari dataset suara pengembang.
- [x] *Fine-tuning* model `SpeechT5` dengan target 150 *steps*.
- [x] Integrasi Vocoder `HiFi-GAN` untuk merender spectrogram menjadi *audio waveform*.

### Fase 4: Integrasi Hibrida & Antarmuka Pengguna (Gradio) 🔄
- [ ] Menginisiasi model `Whisper` sebagai STT Universal pendamping.
- [ ] Membangun UI Gradio dengan struktur **Dual-Tab Architecture**:
  - **TAB 1: Mode Tugas Akhir (Math Focus)**
    - *Alur (Pipeline):* Mic -> STT Wav2Vec2 Kustom -> Middleware Regex -> Cetak UI "1+1=2" -> TTS SpeechT5 Kustom -> Speaker.
    - *Fungsi:* Mendemonstrasikan akurasi sistem yang dilatih *from scratch* untuk mengenali suara pengembang mengucapkan matematika.
  - **TAB 2: Mode Voice Cloning Bebas (Playground)**
    - *Alur (Pipeline):* Mic -> STT Whisper Universal -> Cetak UI Teks Asli -> TTS SpeechT5 Kustom -> Speaker.
    - *Fungsi:* Mendemonstrasikan bahwa "otak mulut" (TTS) berhasil mengkloning suara pengembang dan bisa diimplementasikan pada kalimat sembarang di luar dataset.
- [ ] Menyusun skrip `app.py` yang memuat seluruh *weights* (Wav2Vec2, Whisper, SpeechT5, HiFi-GAN) ke dalam memori/GPU secara asinkron.
 

## 5. Status Terkini
**Update:** 2026-06-01
- Fase 1, 2, dan 3 telah beroperasi 100% secara lokal. Kualitas sintesis TTS menggunakan vocoder HiFi-GAN berhasil mengkloning profil suara dengan baik.
- **Fokus Saat Ini:** Masuk ke Fase 4. Menulis arsitektur *routing* fungsi di Gradio untuk memisahkan *pipeline* antara model Wav2Vec2 dan Whisper sebelum masuk ke *bottleneck* SpeechT5.