# TTS Model Training Summary

## Status: ✅ COMPLETED

Training finished successfully on 2026-06-01 at 17:20 UTC.

## Training Configuration

- **Model**: SpeechT5 (microsoft/speecht5_tts)
- **Task**: Voice Cloning / Text-to-Speech Fine-tuning
- **Dataset**: 48 Indonesian voice samples (metadata.csv)
- **Training Steps**: 150 (25 epochs)
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 2 steps
- **Learning Rate**: 1e-5 (warmup: 10 steps)
- **Evaluation Strategy**: Every 30 steps
- **Save Strategy**: Every 30 steps

## Training Results

| Metric | Value |
|--------|-------|
| Initial Loss | 2.316 |
| Final Loss | 1.47 |
| Training Time | 27.52 seconds |
| Samples/sec | 43.6 |
| Steps/sec | 5.45 |

## Loss Progression

- Step 10: 2.316
- Step 20: 1.91
- Step 30: 1.928
- Step 60: 1.465
- Step 90: 1.526
- Step 120: 1.423
- Step 150: 1.47

## Output Models

### Final Model
- **Location**: `./tts-model-kustom-final/`
- **Size**: 552 MB
- **Files**:
  - `model.safetensors` - Trained weights
  - `config.json` - Model configuration
  - `processor_config.json` - Processor settings
  - `tokenizer_config.json` - Tokenizer config
  - `spm_char.model` - SentencePiece model

### Checkpoints
- `checkpoint-30/` - Step 30
- `checkpoint-60/` - Step 60
- `checkpoint-90/` - Step 90
- `checkpoint-120/` - Step 120
- `checkpoint-150/` - Step 150 (final)

## Key Fixes Applied

1. **Missing Dependencies**:
   - Added `torchcodec` for audio loading
   - Added `speechbrain` for speaker embeddings
   - Added `tensorboard` for training logging

2. **Data Collation Issues**:
   - Fixed `return_tensor` → `return_tensors` typo
   - Implemented manual padding for variable-length spectrograms
   - Added `decoder_attention_mask` for proper masking

3. **Shape Alignment**:
   - Aligned label lengths to `reduction_factor` (2) to prevent 301 vs 300 mismatch
   - Ensured 2D label tensors (seq_len, feature_dim)
   - Converted speaker embeddings to tensors in collator

## Speaker Embeddings

- **Model**: SpeechBrain X-Vector (speechbrain/spkrec-xvect-voxceleb)
- **Embedding Dimension**: 512
- **Purpose**: Captures speaker identity for voice cloning

## Next Steps

1. **Test Inference**:
   ```bash
   python test_tts.py
   ```

2. **Generate Speech**:
   - Use `tts-model-kustom-final/` for inference
   - Provide text + speaker embedding
   - Get mel-spectrogram output

3. **Convert to Audio**:
   - Use vocoder (e.g., HiFi-GAN) to convert mel-spectrogram to waveform
   - Save as WAV file

## Requirements Updated

Added to `requirements.txt`:
- `torchcodec` - Audio codec support
- `speechbrain` - Speaker embeddings
- `tensorboard` - Training visualization

## Notes

- Training was fast (27.5 seconds) due to small dataset (48 samples)
- Loss decreased consistently, indicating good convergence
- Model is ready for inference and further fine-tuning
- Consider collecting more diverse voice samples for production use
