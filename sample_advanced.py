#!/usr/bin/env python
import os
import torch
import torchaudio
import logging

# To enable all DEBUG logging:
# logging.basicConfig(level=logging.DEBUG)
# Or enable only phonemizer DEBUG logging:
# logging.getLogger("phonemizer").setLevel(logging.DEBUG)

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def load_prefix_audio(model, audio_path):
    """
    Loads and preprocesses the prefix audio (expected to be 100ms of silence).
    Returns the encoded prefix codes.
    """
    wav, sr = torchaudio.load(audio_path)
    # Convert to mono by averaging channels if necessary.
    wav = wav.mean(dim=0, keepdim=True)
    wav = model.autoencoder.preprocess(wav, sr)
    wav = wav.to(device, dtype=torch.float32)
    # Add batch dimension before encoding.
    return model.autoencoder.encode(wav.unsqueeze(0))

def main():
    # Load the Zonos model.
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    print("  -> Model loaded.")
    
    # Load reference speaker audio to generate speaker embedding.
    print("Loading reference audio...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reference_audio_path = os.path.join(script_dir, "assets", "exampleaudio.mp3")
    wav, sr = torchaudio.load(reference_audio_path)
    speaker_embedding = model.make_speaker_embedding(wav, sr)
    print("  -> Speaker embedding generated.")

    # Load the prefix audio (100ms silence).
    print("Loading prefix audio...")
    prefix_audio_path = os.path.join(script_dir, "assets", "silence_100ms.wav")
    prefix_audio_codes = load_prefix_audio(model, prefix_audio_path)
    print(" -> Prefix audio loaded.")
    
    # Create and explicitly set the conditioning dictionary.
    # Note: make_cond_dict initializes with basic keys.
    cond_dict = make_cond_dict(
        text="Hello from Zonos, the state of the art text to speech model... ...",
        speaker=speaker_embedding,
        language="en-us",
        emotion=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], # [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
        fmax=22050.0,          # 44.1 kHz output frequency (also support 24000 for 48 kHz)
        pitch_std=45.0,        # Standard deviation for pitch variation
        speaking_rate=15.0,    # Speaking rate in phonemes per minute
        vqscore_8=[0.78] * 8,  # Target VoiceQualityScore for each 1/8th of the audio
        ctc_loss=0.0,          # CTC target loss
        dnsmos_ovrl=4.0,       # Overall DNSMOS score
        speaker_noised=False,  # Speaker noising disabled
        unconditional_keys={"emotion"} # "vqscore_8", "dnsmos_ovrl"
    )
    
    # Prepare conditioning for the model.
    conditioning = model.prepare_conditioning(cond_dict)
    
    # Define generation parameters.
    generation_params = {
        "max_new_tokens": 86 * 30,
        "cfg_scale": 2.0,
        "batch_size": 1,
        "sampling_params": {
            "top_p": 0, 
            "top_k": 0,
            "min_p": 0,
            "linear": 0.5,
            "conf": 0.4,
            "quad": 0.0,
            "repetition_penalty": 3.0,
            "repetition_penalty_window": 2,
            "temperature": 1.0,
        },
        "progress_bar": True,
    }
    
    # For reproducibility.
    torch.manual_seed(421)
    
    # Generate audio codes with the provided prefix audio and generation parameters.
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=prefix_audio_codes,
        **generation_params
    )
    
    # Decode the generated codes to a waveform.
    wavs = model.autoencoder.decode(codes).cpu()
    
    # Save the generated audio.
    output_path = os.path.join(script_dir, "sample_advanced.wav")
    torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
    print(f"Generated audio saved to {output_path}")

if __name__ == "__main__":
    main()
