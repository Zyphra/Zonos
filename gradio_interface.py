import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import torchaudio
import gradio as gr
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from io import BytesIO
from os import getenv
from fastapi.responses import StreamingResponse

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
# Initialize FastAPI
app = FastAPI(title="Zonos API", description="API for Zonos text-to-speech model")

# Keep existing model management code
device = "cuda"
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

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

# API Models
class EmotionConfig(BaseModel):
    happiness: float = Field(1.0, ge=0.0, le=1.0)
    sadness: float = Field(0.05, ge=0.0, le=1.0)
    disgust: float = Field(0.05, ge=0.0, le=1.0)
    fear: float = Field(0.05, ge=0.0, le=1.0)
    surprise: float = Field(0.05, ge=0.0, le=1.0)
    anger: float = Field(0.05, ge=0.0, le=1.0)
    other: float = Field(0.1, ge=0.0, le=1.0)
    neutral: float = Field(0.2, ge=0.0, le=1.0)

class GenerateRequest(BaseModel):
    model_choice: str = Field("Zyphra/Zonos-v0.1-transformer", description="Model variant to use")
    text: str = Field(..., max_length=500, description="Text to synthesize")
    language: str = Field("en-us", description="Language code")
    dnsmos_ovrl: float = Field(4.0, ge=1.0, le=5.0)
    fmax: float = Field(24000, ge=0, le=24000)
    vq_score: float = Field(0.78, ge=0.5, le=0.8)
    pitch_std: float = Field(45.0, ge=0.0, le=300.0)
    speaking_rate: float = Field(15.0, ge=5.0, le=30.0)
    cfg_scale: float = Field(2.0, ge=1.0, le=5.0)
    min_p: float = Field(0.15, ge=0.0, le=1.0)
    seed: int = Field(420)
    emotion: Optional[EmotionConfig] = None
    unconditional_keys: List[str] = Field(["emotion"], description="Conditioning values to ignore")

# API Endpoints
@app.get("/models")
async def get_models():
    """Get available models and their configurations"""
    return {
        "available_models": [
            "Zyphra/Zonos-v0.1-transformer",
            "Zyphra/Zonos-v0.1-hybrid"
        ],
        "supported_languages": supported_language_codes
    }

@app.post("/generate")
async def generate_speech(request: GenerateRequest):
    try:
        model = load_model_if_needed(request.model_choice)

        if request.emotion and "emotion" not in request.unconditional_keys:
            emotion = [
                request.emotion.happiness,
                request.emotion.sadness,
                request.emotion.disgust,
                request.emotion.fear,
                request.emotion.surprise,
                request.emotion.anger,
                request.emotion.other,
                request.emotion.neutral
            ]
            emotion_tensor = torch.tensor(emotion, device=device).unsqueeze(0)
        else:
            emotion_tensor = None

        vq_tensor = torch.tensor([[request.vq_score] * 8], device=device)

        cond_dict = make_cond_dict(
            text=request.text,
            language=request.language,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            device=device,
            unconditional_keys=request.unconditional_keys
        )
        conditioning = model.prepare_conditioning(cond_dict)

        torch.manual_seed(request.seed)
        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=request.cfg_scale,
            batch_size=1,
            sampling_params=dict(min_p=request.min_p)
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        # Ensure we have a 2D tensor for saving (channels, samples)
        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()  # Remove any extra dimensions
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)  # Add channel dimension if needed

        # Now wav_out should be [channels, samples]
        wav_data = BytesIO()
        torchaudio.save(wav_data, wav_out, sr_out, format="wav")
        wav_data.seek(0)

        return StreamingResponse(wav_data, media_type="audio/wav")

    except Exception as e:
        print(f"Error details: {str(e)}")
        print("Tensor shape before save:", wav_out.shape)  # Debug info
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speaker_embedding")
async def create_speaker_embedding(
    file: UploadFile = File(...),
    model_choice: str = "Zyphra/Zonos-v0.1-transformer"
):
    """Generate speaker embedding from audio file"""
    try:
        model = load_model_if_needed(model_choice)

        # Read audio file
        content = await file.read()
        audio_data = BytesIO(content)
        wav, sr = torchaudio.load(audio_data)

        # Generate embedding
        speaker_embedding = model.make_speaker_embedding(wav, sr)

        # Convert to list for JSON response
        embedding_list = speaker_embedding.cpu().numpy().tolist()

        return {"speaker_embedding": embedding_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    min_p,
    seed,
    randomize_seed,
    unconditional_keys,
    progress=gr.Progress(),
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    min_p = float(min_p)
    seed = int(seed)
    max_new_tokens = 86 * 30

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)

    speaker_embedding = None
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        wav, sr = torchaudio.load(speaker_audio)
        speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        with torch.autocast(device, dtype=torch.float32):
            audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
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

    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * 86)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True

    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=min_p),
        callback=update_progress,
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    return (sr_out, wav_out.squeeze().numpy()), seed


def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"],
                    value="Zyphra/Zonos-v0.1-transformer",
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                    max_length=500,  # approximately
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                min_p_slider = gr.Slider(0.0, 1.0, 0.15, 0.01, label="Min P")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                min_p_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
            ],
            outputs=[output_audio, seed_number]
        )

    return demo


# Server run functions defined at module level
def run_gradio():
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)

def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Modified main to run both FastAPI and Gradio
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    from multiprocessing import Process

    # Run both servers in separate processes
    gradio_process = Process(target=run_gradio)
    fastapi_process = Process(target=run_fastapi)

    gradio_process.start()
    fastapi_process.start()

    gradio_process.join()
    fastapi_process.join()
