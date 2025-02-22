import tempfile
import torch
import torchaudio
import gradio as gr
from os import getenv
from utils import process_file, float32_to_int16
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None


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


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    file_update = gr.update(visible=("espeak" in cond_names))
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
        file_update,
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
    file,
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
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
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
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 30

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        # Handle input content
        content_path = None
        if file is not None:
            content_path = file
        elif text.strip():
            # Save text to temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                f.write(text)
                content_path = f.name

        if not content_path:
            raise gr.Error("Please provide either text input or upload a document file")

        if speaker_audio is not None and "speaker" not in unconditional_keys:
            if speaker_audio != SPEAKER_AUDIO_PATH:
                print("Recomputed speaker embedding")
                wav, sr = torchaudio.load(speaker_audio)
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

        emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

        vq_val = float(vq_single)
        vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

        text_chunks = process_file(content_path, 35)  # hard coded for now, could add a slider
        total_chunks = len(text_chunks)
        gr.Info(f"Text split into {total_chunks} chunks. Starting synthesis...")

        audio_segments = []
        # Chunked Generation (unlimited content, yay!)
        for idx, chunk in enumerate(text_chunks, 1):
            if not chunk.strip():
                continue
            gr.Info(f"Generating chunk {idx}/{total_chunks}...")

            estimated_generation_duration = 30 * len(chunk) / 400
            estimated_total_steps = int(estimated_generation_duration * 86)

            def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
                progress((step, estimated_total_steps))
                return True

            cond_dict = make_cond_dict(
                text=chunk.strip(),
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

            codes = selected_model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params=dict(
                    top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic
                ),
                callback=update_progress,
            )

            wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
            sr_out = selected_model.autoencoder.sampling_rate
            if wav_out.dim() == 3:
                wav_out = wav_out.squeeze(0)
            if wav_out.dim() == 1:
                wav_out = wav_out.unsqueeze(0)
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]

            audio_segments.append(wav_out)
            if idx != total_chunks:  # add sub-second silence after every audio chunk except last one
                audio_segments.append(torch.zeros(1, int(sr_out * 0.2)))  # 0.2 seem to be perfect

        if audio_segments:  # most probably
            final_audio = torch.cat(audio_segments, dim=1)  # 1, k
            wav_array = float32_to_int16(final_audio.squeeze().numpy())  # gradio audio except 16bit int array
            return (sr_out, wav_array), seed
        else:
            return (None, None), seed

    except Exception as e:
        gr.Error(str(e))
        return (None, None), seed


def build_interface():
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text Input"):
                    text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=5,
                        max_length=500,
                    )
                with gr.Tab("File Upload"):
                    file = gr.File(label="Upload Document", file_types=[".txt", ".pdf", ".xlsx", ".docx"])
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
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
                    label="Optional Speaker Audio Sample (for cloning)",
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
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(
                        -2.0,
                        2.0,
                        0.5,
                        0.01,
                        label="Linear (set to 0 to disable unified sampling)",
                        info="High values make the output less random.",
                    )
                    # Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(
                        -2.0, 2.0, 0.40, 0.01, label="Confidence", info="Low values make random outputs more random."
                    )
                    quadratic_slider = gr.Slider(
                        -2.0, 2.0, 0.00, 0.01, label="Quadratic", info="High values make low probablities much lower."
                    )
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

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
                happiness = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                sadness = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                disgust = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                fear = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                surprise = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                anger = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                other = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                neutral = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                file,
                language,
                speaker_audio,
                prefix_audio,
                happiness,
                sadness,
                disgust,
                fear,
                surprise,
                anger,
                other,
                neutral,
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
                file,
                language,
                speaker_audio,
                prefix_audio,
                happiness,
                sadness,
                disgust,
                fear,
                surprise,
                anger,
                other,
                neutral,
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
                file,
                language,
                speaker_audio,
                prefix_audio,
                happiness,
                sadness,
                disgust,
                fear,
                surprise,
                anger,
                other,
                neutral,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
