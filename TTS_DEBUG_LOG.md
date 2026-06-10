# TTS Voice Cloning Project - Debug Log and Solutions

## Overview
This project aims to build a custom voice cloning system using SpeechT5 and integrate it into an end-to-end STT-TTS pipeline with Gradio for the PTU Final Project.

## Issues Encountered and Solutions

### 1. Initial Setup and Dependencies
- **Problem**: Missing `datasets` module.
  **Solution**: Installed `datasets` via pip.
- **Problem**: `torchaudio.load` requiring `torchcodec`.
  **Solution**: Installed `torchcodec`.
- **Problem**: Missing `speechbrain` for speaker embedding.
  **Solution**: Installed `speechbrain`.
- **Problem**: Missing `tensorboard` for logging.
  **Solution**: Installed `tensorboard`.

### 2. Tokenizer Issues in SpeechT5
- **Problem**: `return_tensor` typo in `processor` call.
  **Solution**: Changed `return_tensor` to `return_tensors`.
- **Problem**: SpeechT5 tokenizer does not recognize mathematical symbols (+, =, x, :, -).
  **Solution**: Before passing text to TTS, convert symbols to words:
        '+' -> ' ditambah '
        '=' -> ' sama dengan '
        'x' -> ' dikali '
        ':' -> ' dibagi '
        '-' -> ' dikurangi '
     Also map digits to Indonesian words.

### 3. Data Collator and Padding Issues
- **Problem**: Inhomogeneous shapes in label tensors causing `ValueError` in data collator.
  **Solution**: Rewrote the `TTSDataCollator` to manually pad label tensors (mel spectrograms) to the same length, ensuring the length is a multiple of the model's reduction factor (default 2).
- **Problem**: Speaker embeddings as list of lists causing `TypeError` in `torch.stack`.
  **Solution**: Converted each speaker embedding to a tensor before stacking.
- **Problem**: Shape mismatch between model output and labels (e.g., 301 vs 300 frames).
  **Solution**: Adjusted label padding to be a multiple of the reduction factor.

### 4. Device Mismatch
- **Problem**: Tensors on different devices (CPU vs CUDA) causing runtime error.
  **Solution**: Ensured all tensors (input IDs, speaker embeddings) are moved to the same device as the model.

### 5. Audio Output Issues
- **Problem**: Generated audio was silent or extremely low volume.
  **Causes and Solutions**:
  a. **Speaker Embedding Extraction**: 
       - Initially used a very short audio clip (1 word: "satu") for speaker embedding, which lacked sufficient phonetic variation.
       - Fixed by using a longer audio clip (full sentence: `kalimatmtk1.wav`).
  b. **Model Output Silence**:
       - Observed that the model output had several seconds of silence at the beginning before actual speech.
       - Fixed by trimming leading silence based on RMS energy per frame (20ms frames, threshold 0.005).
  c. **Audio Normalization**:
       - The raw output from the vocoder had varying amplitudes, sometimes very low.
       - Fixed by normalizing the audio to the range [-1, 1] after silence trimming.
- **Problem**: Audio sounded like noise, static, or robotic.
  **Causes and Solutions**:
  a. **Incorrect Speaker Embedding**:
       - Fixed by ensuring the embedding is extracted from a clean, representative audio clip and that the tensor operations (squeeze, unsqueeze) are correct.
  b. **Vocoder Issues**:
       - The official `SpeechT5HifiGan` from Hugging Face was used, which is compatible with SpeechT5.
       - Warning about model type mismatch is non-fatal and can be ignored.
  c. **Insufficient Training**:
       - The model was trained for only 150 steps (25 epochs) on a small dataset (48 samples).
       - While this produced audible speech, the quality is limited. For better quality, more training steps and data are needed.

## Current Status
- The system now produces audible speech with the cloned voice.
- The audio is not perfect (some noise and robotic artifacts remain) due to:
      * Limited training data and steps.
      * The inherent limitations of the SpeechT5 model and HiFi-GAN vocoder when fine-tuned on a small dataset.
- The pipeline works end-to-end in the Gradio interface with two tabs:
      1. **Math Focus**: Uses custom Wav2Vec2 STT for math-specific transcription.
      2. **Voice Cloning Playground**: Uses Whisper for universal transcription, then TTS with voice cloning.

## Files Modified
- `train_tts.py`: The training script with fixed data collator, dependency installs, and tokenizer fixes.
- `test_tts_speecht5.py`: Inference script for testing the TTS model in isolation.
- `app.py`: The Gradio application that integrates STT, TTS, and the middleware.
- `requirements.txt`: Updated to include all necessary dependencies.
- `PRD.md`: Updated to reflect completion of Phase 3 (TTS voice cloning).

## Recommendations for Improvement
1. **Increase Training Data**: Collect more voice samples (minimum 1-2 hours) for better speaker embedding and TTS quality.
2. **Increase Training Steps**: Train for more epochs (e.g., 100+ epochs) to reduce loss further.
3. **Use a Pretrained Vocoder**: Consider training a vocoder specifically on the target speaker's data, or use a high-quality pretrained one (we are already using the official HiFi-GAN).
4. **Post-Processing**: Apply noise reduction or equalization to the output audio if needed.
5. **Alternative Models**: Consider using VoxCPM2 (as suggested) for state-of-the-art voice cloning quality, which is tokenizer-free and supports 30 languages including Indonesian.

## Example Code Snippets

### Fixed TTS Inference with Symbol Conversion and Normalization
```python
def generate_custom_tts(text):
    # Convert symbols to words for SpeechT5 compatibility
    text_for_tts = text.replace('+', ' ditambah ').replace('=', ' sama dengan ')
    text_for_tts = text_for_tts.replace('x', ' dikali ').replace(':', ' dibagi ').replace('-', ' dikurangi ')
    num_map = {'1':'satu','2':'dua','3':'tiga','4':'empat','5':'lima','6':'enam','7':'tujuh','8':'delapan','9':'sembilan','0':'nol'}
    for n, word in num_map.items():
        text_for_tts = text_for_tts.replace(n, f" {word} ")
    text_for_tts = " ".join(text_for_tts.split())

    inputs = processor_tts(text=text_for_tts, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        speech = model_tts.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=default_embedding.unsqueeze(0),
            vocoder=vocoder,
        )

    # Post-processing: trim silence and normalize
    speech_np = speech.detach().cpu().numpy()
    frame_size = int(16000 * 0.02)  # 20ms
    trimmed_start = 0
    for i in range(0, len(speech_np) - frame_size, frame_size):
        frame = speech_np[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > 0.005:
            trimmed_start = max(0, i - frame_size)
            break
    speech_np = speech_np[trimmed_start:]

    max_val = np.abs(speech_np).max()
    if max_val > 0:
        speech_np = speech_np / max_val

    return (16000, speech_np)
```

### Speaker Embedding Extraction (Fixed)
```python
ref_audio = "voice-dataset/kalimatmtk/kalimatmtk1.wav"
audio, sr = torchaudio.load(ref_audio)
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)
if sr != 16000:
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
audio = audio.squeeze()  # to 1D

with torch.no_grad():
    speaker_embeddings = spk_model.encode_batch(audio.unsqueeze(0).to(DEVICE))
    speaker_embeddings = speaker_embeddings.squeeze(0).squeeze(0)
```

## Conclusion
Despite the limitations, the system successfully demonstrates voice cloning and end-to-end STT-TTS functionality.
For production use, consider the recommendations above or explore advanced models like VoxCPM2.
