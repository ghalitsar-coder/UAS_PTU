# Product Requirements Document (PRD)
**Proyek:** Sistem Kustom Speech-to-Text (STT) & Text-to-Speech (TTS) Terintegrasi
**Mata Kuliah:** Tugas Akhir Pengantar Teknologi Informasi (PTU) - Proyek Kelompok
**Lead Developer / Engineer:** Galitsar Gyasi Elfaris
**Institusi:** Institut Teknologi Nasional (ITENAS), Bandung

## 1. Ringkasan Eksekutif
Proyek ini bertujuan untuk membangun sebuah sistem kecerdasan buatan (*Artificial Intelligence*) *end-to-end* yang mampu memproses input suara pengguna, mentranskripsikannya menjadi teks (termasuk konversi simbol matematika), dan menyuarakannya kembali menggunakan model *voice cloning* dengan suara kustom (suara pengembang).

## 2. Tujuan dan Target Keluaran
* **Akurasi Transkripsi:** Model STT dapat mengenali kalimat umum dan operasi matematika bahasa Indonesia.
* **Pemrosesan Konteks:** Sistem harus mampu merapikan teks (misal: "tambah" menjadi `+`, "sama dengan" menjadi `=`).
* **Kloning Suara:** Model TTS harus bisa mensintesis teks kembali menjadi audio yang natural dengan karakteristik suara kustom.
* **Target Output Uji Coba:** Sistem berhasil memproses dan menampilkan teks: `"1+1=2 1+2=3 3+1=4 123 sayang semuanya"`.

## 3. Arsitektur Sistem & Teknologi
* **Input/Output:** File Audio `.wav` (Mono, 16kHz).
* **Model STT (Pendengaran):** `Wav2Vec2-large-xlsr-indonesian` (di-*fine-tune* menggunakan Hugging Face `transformers` & `datasets`).
* **Pemrosesan Teks (Middleware):** Skrip Python (Regex/Dictionary Mapping) untuk konversi simbol matematis.
* **Model TTS (Sintesis Suara):** `Supertonic-3` (menggunakan ONNX Runtime untuk inferensi lokal yang efisien).
* **Antarmuka (UI):** Gradio / Streamlit.
* **Rencana Infrastruktur (DevOps):** Kontainerisasi dengan Docker, direncanakan untuk *deployment* menggunakan Kubernetes di layanan *cloud* (Azure/AWS) sebagai implementasi *backend* modern.

## 4. Fase Implementasi dan Status (Checklist)

### Fase 1: Persiapan Dataset (Data Collection)
- [x] Merekam suara untuk huruf (A-Z) dan angka (0-9) secara terisolasi.
- [x] Merekam frasa target dan variasi operasi matematika secara natural.
- [x] Merekam kalimat fonetik umum bahasa Indonesia.
- [x] Konversi seluruh format audio dari `.m4a` ke `.wav` (Mono, 16kHz) menggunakan FFmpeg.
- [x] Menyusun file transkripsi `metadata.csv` (teks *lowercase*, tanpa tanda baca).
- [x] Validasi dataset untuk memastikan *array* audio terbaca sempurna oleh model.

### Fase 2: Pengembangan STT (Wav2Vec2)
- [x] Inisiasi lingkungan Python dan instalasi *library* inti (`transformers`, `datasets`, `librosa`, dll).
- [x] *Preprocessing* dataset (ekstraksi fitur audio ke `input_values` dan tokenisasi teks ke `labels`).
- [x] Mengunduh *weights* model dasar `indonesian-nlp/wav2vec2-large-xlsr-indonesian` (1.26GB).
- [x] Penyesuaian arsitektur model (mengatasi *size mismatch* pada *layer* klasifikasi untuk 30 token unik).
- [x] *Patching* pembaruan API Hugging Face (mengganti fungsi `extractor` ke `encoder` dan perbaikan `TrainingArguments`).
- [ ] Menyelesaikan perbaikan *library* (`pip install accelerate>=1.1.0`).
- [ ] Mengeksekusi proses *Fine-Tuning* (Pelatihan Model) dengan `Trainer` API.
- [ ] Menyimpan model STT final (`./stt-model-kustom-final`).

### Fase 3: Pengembangan TTS (Voice Cloning)
- [ ] Mempelajari struktur *input* yang dibutuhkan oleh model `Supertonic-3`.
- [ ] Mempersiapkan *Voice Style JSON* atau memproses referensi audio untuk kloning suara lokal.
- [ ] Menguji coba sintesis teks ke suara menggunakan Python SDK `supertonic`.

### Fase 4: Integrasi & Antarmuka Pengguna
- [ ] Membuat fungsi *post-processing* Python untuk merapikan teks ("satu tambah satu" -> "1+1").
- [ ] Membangun alur *pipeline* berurutan: Mic -> STT -> *Post-processing* teks -> TTS -> Speaker.
- [ ] Membangun UI sederhana berbasis web menggunakan Gradio/Streamlit.

### Fase 5: Deployment (Opsional/Pengembangan Lanjut)
- [ ] Membuat `Dockerfile` untuk membungkus seluruh aplikasi, model, dan dependensi.
- [ ] Mengonfigurasi Kubernetes manifest (`deployment.yaml`, `service.yaml`) jika akan diunggah ke *cluster*.
- [ ] *Push* ke GitLab / GitHub repository.