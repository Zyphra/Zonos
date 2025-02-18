import torch
import torchaudio
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    model.requires_grad_(False).eval()

    # Load a reference speaker audio to generate a speaker embedding.
    print("Loading reference audio...")
    wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(421)

    # Define the text prompt.
    text = "Hello, world! This is a test of streaming generation from Zonos."
    
    # Create the conditioning dictionary (using text, speaker embedding, language, etc.).
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)

    # --- STREAMING GENERATION ---
    # We set max_new_tokens lower for a short test (adjust as needed).
    chunk_size = 40          # yield every 40 generated tokens, less for faster streaming but lower quality

    print("Starting streaming generation...")
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,  # no audio prefix in this test
        chunk_size=chunk_size
    )

    # Accumulate audio chunks as they are generated.
    audio_chunks = []
    for sr_out, codes_chunk in stream_generator:
        print(f"Received codes chunk of shape: {codes_chunk.shape}")
        audio_chunk = model.autoencoder.decode(codes_chunk).cpu()
        audio_chunks.append(audio_chunk[0])

    if len(audio_chunks) == 0:
        print("No audio chunks were generated.")
        return

    # Concatenate all audio chunks along the time axis.
    full_audio = np.concatenate(audio_chunks, axis=-1)
    out_sr = model.autoencoder.sampling_rate

    # Save the full audio as a WAV file.
    out_tensor = torch.tensor(full_audio)
    torchaudio.save("stream_sample.wav", out_tensor, out_sr)
    print("Saved streaming audio to 'stream_sample.wav'.")

if __name__ == "__main__":
    main()
