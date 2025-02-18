import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os
from tqdm import tqdm
from datasets import load_dataset
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

class EmotionEncoder:
    """Encodes emotions into embeddings using a predefined emotion set."""
    def __init__(self):
        self.emotions = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fearful": 4,
            "disgust": 5,
            "surprised": 6
        }
        self.emotion_embeddings = torch.nn.Embedding(len(self.emotions), 256)  # 256-dim emotion embedding
    
    def encode(self, emotion: str) -> torch.Tensor:
        if emotion not in self.emotions:
            emotion = "neutral"  # default to neutral if emotion not found
        emotion_idx = torch.tensor([self.emotions[emotion]])
        return self.emotion_embeddings(emotion_idx)

class AudioPreprocessor:
    """Handles audio preprocessing including normalization and augmentation."""
    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        
        # Mel spectrogram parameters
        self.n_fft = 1024
        self.win_length = int(0.025 * sample_rate)  # 25ms window
        self.hop_length = int(0.01 * sample_rate)   # 10ms hop
        self.n_mels = 80
        
        # Calculate maximum mel length to ensure consistent sizes
        self.max_mel_length = (self.max_length - self.n_fft) // self.hop_length + 3
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=8000,
            window_fn=torch.hann_window,
            normalized=True
        )
        
    def process(self, wav: torch.Tensor, augment: bool = True) -> torch.Tensor:
        """Process audio and return processed audio."""
        # Ensure audio is 2D (channels, time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
            
        # Ensure consistent length
        if wav.size(-1) > self.max_length:
            start = 0  # Always take from start for consistency
            wav = wav[..., start:start + self.max_length]
        else:
            # Pad with zeros if too short
            pad_length = self.max_length - wav.size(-1)
            wav = torch.nn.functional.pad(wav, (0, pad_length))
            
        # Normalize audio to [-1, 1]
        wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
        
        if augment:
            # Random volume adjustment
            wav = wav * (0.8 + 0.4 * torch.rand(1))
            
            # Random noise addition (SNR between 20-30dB)
            if torch.rand(1) < 0.5:
                noise_level = 10 ** (-torch.rand(1) * 10 - 20)  # -20 to -30 dB
                noise = torch.randn_like(wav) * noise_level
                wav = wav + noise
        
        return wav

class ZonosHFDataset(Dataset):
    def __init__(
        self, 
        dataset_name="mozilla-foundation/common_voice_17_0", 
        language="uz", 
        split="train", 
        sampling_rate=16000,
        emotion_labels_path: Optional[str] = None
    ):
        self.sampling_rate = sampling_rate
        self.language = language
        self.audio_processor = AudioPreprocessor(sampling_rate)
        
        print(f"Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name, language, split=split)
        print(f"Dataset loaded with {len(self.dataset)} samples")
        
        # Load emotion labels if provided
        self.emotion_labels = {}
        if emotion_labels_path and os.path.exists(emotion_labels_path):
            with open(emotion_labels_path, 'r') as f:
                self.emotion_labels = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def get_emotion(self, idx: int) -> List[float]:
        """Get emotion vector for a sample, defaulting to neutral if not found."""
        if str(idx) in self.emotion_labels:
            return self.emotion_labels[str(idx)]
        return [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]  # default neutral
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process audio
        audio = item['audio']
        wav = torch.FloatTensor(audio['array'])
        sr = audio['sampling_rate']
        
        # Ensure 2D (channels, time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        
        # Ensure fixed length (30 seconds)
        target_length = 30 * self.sampling_rate
        if wav.size(-1) > target_length:
            wav = wav[..., :target_length]
        else:
            # Pad with zeros if too short
            pad_length = target_length - wav.size(-1)
            wav = torch.nn.functional.pad(wav, (0, pad_length))
        
        # Process audio
        wav = self.audio_processor.process(wav)
        
        # Get emotion vector
        emotion = torch.tensor(self.get_emotion(idx), dtype=torch.float32)
        
        return {
            'audio': wav.squeeze(0),  # Remove channel dimension for speaker model
            'text': item['sentence'],
            'language': self.language,
            'emotion': emotion
        }

def train_zonos(
    model_path="Zyphra/Zonos-v0.1-transformer",
    dataset_name="mozilla-foundation/common_voice_17_0",
    language="uz",
    output_dir="checkpoints",
    batch_size=4,  # Reduced batch size
    learning_rate=1e-4,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    emotion_labels_path: Optional[str] = None
):
    # Initialize model
    model = Zonos.from_pretrained(model_path, device=device)
    model.train()
    
    # Create dataset and dataloader
    dataset = ZonosHFDataset(
        dataset_name, 
        language, 
        emotion_labels_path=emotion_labels_path
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced number of workers
        collate_fn=lambda x: {
            'audio': torch.stack([s['audio'] for s in x]),  # Audio is already properly shaped
            'text': [s['text'] for s in x],
            'language': [s['language'] for s in x],
            'emotion': torch.stack([s['emotion'] for s in x])
        }
    )
    
    # Initialize optimizer with warmup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Move tensors to device
            audio = batch['audio'].to(device)  # Shape: [batch_size, 480000]
            emotions = batch['emotion'].to(device)
            
            # Process each sample in the batch
            batch_loss = 0
            for i in range(len(audio)):
                # Create speaker embedding from the audio
                speaker = model.make_speaker_embedding(audio[i], dataset.sampling_rate)  # Shape: [1, 256]
                
                # Prepare conditioning with emotion vector
                cond_dict = make_cond_dict(
                    text=batch['text'][i],
                    speaker=speaker,
                    language=batch['language'][i],
                    emotion=emotions[i].tolist()
                )
                conditioning = model.prepare_conditioning(cond_dict)
                
                # Get target codes using the autoencoder
                # Ensure audio is properly shaped for the autoencoder
                audio_input = audio[i].unsqueeze(0)  # Add batch dimension: [1, 480000]
                if audio_input.dim() == 1:
                    audio_input = audio_input.unsqueeze(0)  # Add batch dimension if needed
                if audio_input.dim() == 2:
                    audio_input = audio_input.unsqueeze(1)  # Add channel dimension if needed
                
                # Resample to 44.1kHz for DAC
                if dataset.sampling_rate != 44100:
                    audio_input = torchaudio.functional.resample(
                        audio_input, 
                        dataset.sampling_rate, 
                        44100
                    )
                
                # Ensure proper padding for DAC
                target_length = int(44100 * 30)  # 30 seconds at 44.1kHz
                if audio_input.size(-1) < target_length:
                    pad_length = target_length - audio_input.size(-1)
                    audio_input = torch.nn.functional.pad(audio_input, (0, pad_length))
                elif audio_input.size(-1) > target_length:
                    audio_input = audio_input[..., :target_length]
                
                with torch.no_grad():
                    target_codes = model.autoencoder.encode(audio_input)  # Shape: [1, num_codebooks, seq_len]
                    target_codes = target_codes.to(device)
                
                # Forward pass
                output = model(conditioning)  # Shape: [batch_size, num_codebooks, seq_len, vocab_size]
                
                # Calculate token prediction loss
                # Ensure output and target_codes have compatible shapes
                target_codes = target_codes.squeeze(0)  # Remove batch dimension from target_codes
                
                # Get the minimum sequence length between output and target
                seq_len = min(output.size(2), target_codes.size(-1))
                
                # Truncate both tensors to the same sequence length
                output = output[..., :seq_len, :]  # [batch_size, num_codebooks, seq_len, vocab_size]
                target_codes = target_codes[..., :seq_len]  # [num_codebooks, seq_len]
                
                # Reshape for cross entropy - fixed dimensions for 4D tensor
                output = output.squeeze(0)  # Remove batch dimension: [num_codebooks, seq_len, vocab_size]
                output = output.permute(1, 0, 2)  # [seq_len, num_codebooks, vocab_size]
                output = output.contiguous().view(-1, output.size(-1))  # [seq_len * num_codebooks, vocab_size]
                target_codes = target_codes.transpose(0, 1)  # [seq_len, num_codebooks]
                target_codes = target_codes.reshape(-1)  # [seq_len * num_codebooks]
                
                token_loss = torch.nn.functional.cross_entropy(output, target_codes)
                
                batch_loss += token_loss
            
            # Average loss over batch
            batch_loss = batch_loss / batch_size
            
            # Backward pass
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += batch_loss.item()
        
        # End of epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"zonos_checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start training
    train_zonos(
        model_path="Zyphra/Zonos-v0.1-transformer",
        dataset_name="mozilla-foundation/common_voice_17_0",
        language="uz",
        output_dir="checkpoints",
        batch_size=4,
        num_epochs=10,
        emotion_labels_path="emotion_labels.json"  # Optional path to emotion labels
    ) 