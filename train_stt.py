import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import torch
import librosa
from dataclasses import dataclass
from typing import Dict, List, Union

# 1. Menyiapkan Data Collator untuk padding audio yang panjangnya berbeda
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Memisahkan input_values dan labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Padding input_values (audio)
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Padding labels (teks) menggunakan tokenizer secara langsung (API Baru)
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Mengganti padding token di label menjadi -100 agar diabaikan saat menghitung loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def main():
    print("1. Memuat Processor dan Dataset...")
    processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian")
    dataset = load_dataset('csv', data_files='metadata.csv', split='train')

    # Pemrosesan ulang dataset
    def prepare_dataset(batch):
        audio_array, _ = librosa.load(batch["file_name"], sr=16000)
        # Menambahkan parameter yang disukai linter modern
        batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    print("2. Memproses Dataset (Mohon tunggu sebentar)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print("3. Mengunduh dan Menyiapkan Model Wav2Vec2...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "indonesian-nlp/wav2vec2-large-xlsr-indonesian", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    
    # Menggunakan metode terbaru untuk membekukan encoder
    model.freeze_feature_encoder()

    print("4. Menyiapkan Parameter Pelatihan...")
    training_args = TrainingArguments(
        output_dir="./stt-model-kustom",
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps", # <-- Diperbarui dari evaluation_strategy
        num_train_epochs=10,
        fp16=False,
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=10,
        save_total_limit=2,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor, # <-- Diperbarui dari tokenizer
    )

    print("5. Memulai Pelatihan! 🚀")
    trainer.train()
    
    print("\n✅ Pelatihan selesai! Menyimpan model akhir...")
    trainer.save_model("./stt-model-kustom-final")
    processor.save_pretrained("./stt-model-kustom-final")

if __name__ == "__main__":
    main()