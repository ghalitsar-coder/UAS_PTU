import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
from datasets import load_dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from speechbrain.inference.speaker import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def main():
    print("1. Memuat Processor dan Model SpeechT5...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    
    print("2. Memuat Ekstraktor Karakteristik Suara (SpeechBrain)...")
    # Model ini akan mengenali "warna" suaramu (X-Vector)
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb", 
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        savedir=os.path.join("/tmp", "speechbrain")
    )

    print("3. Memuat dan Memproses Dataset...")
    dataset = load_dataset('csv', data_files='metadata.csv', split='train')

    def prepare_dataset(batch):
        # Memuat audio
        audio, sr = torchaudio.load(batch["file_name"])
        # Konversi ke Mono jika perlu
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample ke 16kHz jika perlu
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
            
        audio = audio.squeeze()

        # Ekstrak fitur Teks dan Audio Log-Mel Spectrogram
        inputs = processor(
            text=batch["transcription"], 
            audio_target=audio.numpy(), 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Ekstrak DNA Suara (Speaker Embedding) dari audio ini
        with torch.no_grad():
            speaker_embeddings = spk_model.encode_batch(audio.unsqueeze(0))
            speaker_embeddings = speaker_embeddings.squeeze(0).squeeze(0)

        batch["input_ids"] = inputs["input_ids"][0]
        batch["labels"] = inputs["labels"][0]
        batch["speaker_embeddings"] = speaker_embeddings
        
        return batch

    # Menerapkan fungsi ke seluruh dataset
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("4. Menyiapkan Parameter Pelatihan TTS...")
    # Data Collator khusus untuk TTS (Seq2Seq)
    @dataclass
    class TTSDataCollator:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]
            label_tensors = [torch.as_tensor(feature["labels"]) for feature in features]

            # Pad input_ids
            batch = self.processor.tokenizer.pad(input_ids, padding=True, return_tensors="pt")

            # SpeechT5 decoder reduces frames by model.config.reduction_factor (default 2).
            # Label length must be a multiple of that factor, otherwise model output/mask
            # can differ by 1 frame (e.g. 151 vs 150).
            reduction_factor = getattr(model.config, "reduction_factor", 2)
            max_label_len = max(t.shape[0] for t in label_tensors)
            if max_label_len % reduction_factor != 0:
                max_label_len += reduction_factor - (max_label_len % reduction_factor)
            # Pad labels: find max length and pad with 0.0
            # The model will handle masking internally
            padded_labels = []
            for t in label_tensors:
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                # Pad with zeros, not -100
                if t.shape[0] < max_label_len:
                    pad_len = max_label_len - t.shape[0]
                    pad_tensor = torch.zeros((pad_len, t.shape[1]), dtype=t.dtype)
                    t = torch.cat([t, pad_tensor], dim=0)
                padded_labels.append(t)

            batch["labels"] = torch.stack(padded_labels)
            batch["speaker_embeddings"] = torch.stack([torch.as_tensor(s) for s in speaker_features])
            
            # Create decoder attention mask for labels
            labels_attention_mask = torch.zeros((len(label_tensors), max_label_len), dtype=torch.long)
            for i, t in enumerate(label_tensors):
                labels_attention_mask[i, :t.shape[0]] = 1
            batch["decoder_attention_mask"] = labels_attention_mask
            
            return batch

    data_collator = TTSDataCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./tts-model-kustom",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5, # LR yang lebih kecil agar suara tidak rusak
        warmup_steps=10,
        max_steps=150,      # Iterasi latih TTS (bisa ditambah jika kurang mirip)
        eval_strategy="steps",
        eval_steps=30,
        save_steps=30,
        logging_steps=10,
        report_to=["tensorboard"],
        # Optimasi performa untuk GPU RTX 4060
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=processor,
    )

    print("5. Memulai Pelatihan Voice Cloning! 🗣️🚀")
    trainer.train()
    
    print("\n✅ Pelatihan selesai! Menyimpan model akhir...")
    trainer.save_model("./tts-model-kustom-final")
    processor.save_pretrained("./tts-model-kustom-final")

if __name__ == "__main__":
    main()