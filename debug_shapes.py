import torch
import torchaudio
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from speechbrain.inference.speaker import EncoderClassifier
import os

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    savedir=os.path.join("/tmp", "speechbrain")
)

dataset = load_dataset('csv', data_files='metadata.csv', split='train')

def prepare_dataset(batch):
    audio, sr = torchaudio.load(batch["file_name"])
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze()
    
    inputs = processor(
        text=batch["transcription"],
        audio_target=audio.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        speaker_embeddings = spk_model.encode_batch(audio.unsqueeze(0))
        speaker_embeddings = speaker_embeddings.squeeze(0).squeeze(0)
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Labels shape: {inputs['labels'].shape}")
    print(f"Speaker embeddings shape: {speaker_embeddings.shape}")
    
    batch["input_ids"] = inputs["input_ids"][0]
    batch["labels"] = inputs["labels"][0]
    batch["speaker_embeddings"] = speaker_embeddings
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
print(f"Dataset features: {dataset.features}")
print(f"First example keys: {dataset[0].keys()}")
for k, v in dataset[0].items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v)}")
