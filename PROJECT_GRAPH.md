# 📊 UAS PTU Project Graph

Berikut adalah visualisasi alur sistem Integrated STT-TTS dengan VoxCPM2.

## 1. System Architecture Pipeline

```mermaid
graph TD
    subgraph "Input Stage"
        A[User Voice Input] -->|Microphone/Upload| B{Mode Selection}
    end

    subgraph "STT Processing"
        B -->|Math Mode| C[Custom STT: Wav2Vec2]
        B -->|Playground Mode| D[Universal STT: Whisper Small]
        
        C -->|Raw Text| E[Math Middleware]
        D -->|Raw Text| F[Direct Text]
    end

    subgraph "Logic & Formatting"
        E -->|Symbolic: 1+1=2| G[Text Normalizer]
        F -->|Clean Text| G
        G -->|Phonetic: satu ditambah satu| H[TTS Input Processor]
    end

    subgraph "TTS Engine (VoxCPM2)"
        I[(Voice Dataset)] -->|kalimatmtk1.wav| J[Voice Cloning Module]
        H --> J
        J -->|CUDA Inference| K[Synthesized Audio Output]
    end

    K --> L[Gradio UI Playback]
```

## 2. Directory Structure & Dependencies

```mermaid
graph LR
    Root[UAS_PTU/] --> App[app_voxcpm.py]
    Root --> Models[pretrained_models/]
    Root --> Dataset[voice-dataset/]
    Root --> CustomSTT[stt-model-kustom-final/]

    Models --> VoxCPM[VoxCPM2/]
    VoxCPM --> Safetensors[model.safetensors]
    
    Dataset --> MTK[kalimatmtk/]
    Dataset --> Angka[angka/]
    Dataset --> Fonetik[fonetik/]
    
    CustomSTT --> W2V[Wav2Vec2 Weights]
```

## 3. Data Flow Detail (Math Mode)

```mermaid
sequenceDiagram
    participant U as User (Audio)
    participant S as STT (Wav2Vec2)
    participant M as Middleware
    participant T as TTS (VoxCPM2)
    participant D as Dataset (Ref)

    U->>S: Kirim Audio (.wav)
    S->>M: "satu tambah satu sama dengan dua"
    M->>M: Convert ke "1+1=2"
    M->>T: Convert balik ke "satu ditambah satu sama dengan dua" (untuk TTS)
    D->>T: Load reference_wav_path
    T->>U: Output Audio Voice Cloned
```
