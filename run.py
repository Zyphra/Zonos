import io
import wave
import base64
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from os import getenv

# Imports from zonos
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# Global variables for model caching and speaker embedding
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None
ROOT_PATH = Path(__file__).parent


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a NumPy array (assumed to be float32 in the range [-1, 1]) to WAV bytes (PCM 16-bit).
    """
    # Determine the number of channels.
    if audio_np.ndim == 1:
        channels = 1
    elif audio_np.ndim == 2:
        channels = audio_np.shape[0]
    else:
        raise ValueError("Invalid audio shape")

    # Convert float values to int16.
    audio_int16 = np.int16(audio_np * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer.read()


def generate_audio_api(
        model_choice: str,
        text: str,
        language: str,
        speaker_audio: Optional[str],
        prefix_audio: Optional[str],
        e1: float,
        e2: float,
        e3: float,
        e4: float,
        e5: float,
        e6: float,
        e7: float,
        e8: float,
        vq_single: float,
        fmax: float,
        pitch_std: float,
        speaking_rate: float,
        dnsmos_ovrl: float,
        speaker_noised: bool,
        cfg_scale: float,
        top_p: float,
        top_k: int,
        min_p: float,
        linear: float,
        confidence: float,
        quadratic: float,
        seed: int,
        randomize_seed: bool,
        unconditional_keys: List[str],
):
    """
    Generates audio based on the provided parameters.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 30  # approximate token count

    global SPEAKER_EMBEDDING, SPEAKER_AUDIO_PATH
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)

    # Compute the speaker embedding if a speaker audio path is provided
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio != SPEAKER_AUDIO_PATH:
            print("Recomputed speaker embedding")
            asset_path = ROOT_PATH / "assets" / speaker_audio
            wav, sr = torchaudio.load(asset_path)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    # Build emotion and VQ conditioning tensors.
    emotion_tensor = torch.tensor([e1, e2, e3, e4, e5, e6, e7, e8], device=device)
    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=SPEAKER_EMBEDDING,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    # Use a dummy callback as we don't track progress in this API.
    def dummy_callback(_frame: torch.Tensor, step: int, total_steps: int) -> bool:
        return True

    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
        callback=dummy_callback,
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    audio_np = wav_out.squeeze().numpy()
    return sr_out, audio_np, seed


# Define Pydantic models for the API request and response

class AudioGenerationRequest(BaseModel):
    model_choice: str = Field(
        "Zyphra/Zonos-v0.1-transformer", description="Zonos Model Type"
    )
    text: str = Field(..., description="Text to Synthesize")
    language: str = Field("en-us", description="Language Code")
    speaker_audio: str = Field(..., description="Path to the cloned voice audio")
    prefix_audio: Optional[str] = Field(
        "assets/silence_100ms.wav", description="Optional prefix audio file path"
    )
    e1: float = Field(1.0, description="Happiness")
    e2: float = Field(0.05, description="Sadness")
    e3: float = Field(0.05, description="Disgust")
    e4: float = Field(0.05, description="Fear")
    e5: float = Field(0.05, description="Surprise")
    e6: float = Field(0.05, description="Anger")
    e7: float = Field(0.1, description="Other")
    e8: float = Field(0.2, description="Neutral")
    vq_single: float = Field(0.78, description="VQ Score")
    fmax: float = Field(24000, description="Fmax (Hz)")
    pitch_std: float = Field(45.0, description="Pitch Std")
    speaking_rate: float = Field(15.0, description="Speaking Rate")
    dnsmos_ovrl: float = Field(4.0, description="DNSMOS Overall")
    speaker_noised: bool = Field(False, description="Denoise Speaker")
    cfg_scale: float = Field(2.0, description="CFG Scale")
    top_p: float = Field(0.0, description="Top P")
    top_k: int = Field(0, description="Top K")
    min_p: float = Field(0.0, description="Min P")
    linear: float = Field(0.5, description="Linear sampling parameter")
    confidence: float = Field(0.40, description="Confidence sampling parameter")
    quadratic: float = Field(0.0, description="Quadratic sampling parameter")
    seed: int = Field(420, description="Seed for generation")
    randomize_seed: bool = Field(True, description="Randomize seed before generation")
    unconditional_keys: List[str] = Field(
        default_factory=lambda: ["emotion"],
        description="List of conditioning keys to treat as unconditional",
    )


class AudioGenerationResponse(BaseModel):
    seed: int
    sample_rate: int
    audio_base64: str  # Base64-encoded WAV file


app = FastAPI(title="Zonos Audio Generation API")


@app.post("/generate_audio", response_model=AudioGenerationResponse)
def generate_audio_endpoint(request: AudioGenerationRequest):
    try:
        sr, audio_np, used_seed = generate_audio_api(
            model_choice=request.model_choice,
            text=request.text,
            language=request.language,
            speaker_audio=request.speaker_audio,
            prefix_audio=request.prefix_audio,
            e1=request.e1,
            e2=request.e2,
            e3=request.e3,
            e4=request.e4,
            e5=request.e5,
            e6=request.e6,
            e7=request.e7,
            e8=request.e8,
            vq_single=request.vq_single,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            speaker_noised=request.speaker_noised,
            cfg_scale=request.cfg_scale,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            linear=request.linear,
            confidence=request.confidence,
            quadratic=request.quadratic,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            unconditional_keys=request.unconditional_keys,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert the generated audio to WAV bytes and then to a base64 string.
    wav_bytes = numpy_to_wav_bytes(audio_np, sr)
    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

    return AudioGenerationResponse(
        seed=used_seed,
        sample_rate=sr,
        audio_base64=audio_base64,
    )


if __name__ == "__main__":
    import uvicorn

    # Optionally, use an environment variable to control sharing.
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    uvicorn.run(app, host="0.0.0.0", port=7860)
