import os
import torch
import torchaudio
import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download
from underthesea import sent_tokenize
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- Load model ---
checkpoint_dir = "model/"
os.makedirs(checkpoint_dir, exist_ok=True)

required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
if not all(f in os.listdir(checkpoint_dir) for f in required_files):
    print("Dang tai model viXTTS...")
    snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=checkpoint_dir)
    hf_hub_download(repo_id="coqui/XTTS-v2", filename="speakers_xtts.pth", local_dir=checkpoint_dir)

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Dang load model...", flush=True)
config = XttsConfig()
config.load_json(os.path.join(checkpoint_dir, "config.json"))
print("Config loaded", flush=True)
model = Xtts.init_from_config(config)
print("Init done, loading checkpoint...", flush=True)
model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=False)
if torch.cuda.is_available():
    model.cuda()
    print("GPU mode", flush=True)
else:
    print("CPU mode", flush=True)
print("Model ready!", flush=True)


def generate_voice(text, voice_file, language, speed, temperature):
    if not text or not text.strip():
        raise gr.Error("Vui lòng nhập nội dung cần đọc!")
    if voice_file is None:
        raise gr.Error("Vui lòng upload file giọng mẫu!")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=voice_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    sentences = sent_tokenize(text)
    wav_chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        wav_chunk = model.inference(
            text=sentence,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
            speed=speed,
            enable_text_splitting=True,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    output_path = "output.wav"
    torchaudio.save(output_path, out_wav, 24000)
    return output_path, output_path


CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5rem;
}
.main-title h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.5rem;
    font-size: 1rem;
}
footer { display: none !important; }
"""

with gr.Blocks() as app:

    gr.HTML('<div class="main-title"><h1>Voice Clone</h1></div>')
    gr.HTML('<div class="subtitle">Clone giọng nói tiếng Việt với AI — Powered by viXTTS</div>')

    with gr.Row(equal_height=True):
        # --- Left column: inputs ---
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Nội dung cần đọc",
                placeholder="Nhập văn bản tiếng Việt tại đây...",
                lines=6,
                max_lines=20,
            )
            voice_input = gr.Audio(
                label="Giọng mẫu (WAV, 6-30 giây)",
                type="filepath",
                sources=["upload", "microphone"],
            )

            with gr.Row():
                lang_input = gr.Dropdown(
                    choices=[
                        ("Tiếng Việt", "vi"),
                        ("English", "en"),
                        ("Chinese", "zh-cn"),
                        ("Japanese", "ja"),
                        ("Korean", "ko"),
                        ("French", "fr"),
                        ("German", "de"),
                        ("Spanish", "es"),
                        ("Portuguese", "pt"),
                        ("Italian", "it"),
                        ("Russian", "ru"),
                        ("Arabic", "ar"),
                        ("Polish", "pl"),
                        ("Turkish", "tr"),
                        ("Dutch", "nl"),
                        ("Czech", "cs"),
                        ("Hungarian", "hu"),
                        ("Hindi", "hi"),
                    ],
                    value="vi",
                    label="Ngôn ngữ",
                )
                speed_input = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Tốc độ",
                )
                temp_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Creativity",
                )

            generate_btn = gr.Button(
                "Tạo giọng nói",
                variant="primary",
                size="lg",
            )

        # --- Right column: output ---
        with gr.Column(scale=2):
            audio_output = gr.Audio(
                label="Kết quả",
                type="filepath",
                interactive=False,
            )
            download_output = gr.File(
                label="Tải về",
            )

            gr.Markdown(
                """
                ### Hướng dẫn
                1. **Upload giọng mẫu** — file WAV/MP3, 6-30 giây, giọng rõ ràng
                2. **Nhập nội dung** — câu dài trên 10 từ cho kết quả tốt nhất
                3. **Bấm "Tạo giọng nói"** — đợi vài phút (CPU) hoặc vài giây (GPU)
                4. **Nghe thử & tải về** kết quả
                """
            )

    generate_btn.click(
        fn=generate_voice,
        inputs=[text_input, voice_input, lang_input, speed_input, temp_input],
        outputs=[audio_output, download_output],
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    )
