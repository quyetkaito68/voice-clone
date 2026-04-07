import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from underthesea import sent_tokenize
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- 1. Download & Load Model ---
checkpoint_dir = "model/"
os.makedirs(checkpoint_dir, exist_ok=True)

required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
if not all(f in os.listdir(checkpoint_dir) for f in required_files):
    print("Dang tai model viXTTS...")
    snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=checkpoint_dir)
    hf_hub_download(repo_id="coqui/XTTS-v2", filename="speakers_xtts.pth", local_dir=checkpoint_dir)

print("Dang load model...")
config = XttsConfig()
config.load_json(os.path.join(checkpoint_dir, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=False)
if torch.cuda.is_available():
    model.cuda()
    print("Dang chay tren GPU")
else:
    print("Dang chay tren CPU")

# --- 2. Speaker embedding tu audio mau ---
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path="voice.wav",
    gpt_cond_len=model.config.gpt_cond_len,
    max_ref_length=model.config.max_ref_len,
    sound_norm_refs=model.config.sound_norm_refs,
)

text = "Câu hỏi này rất thực tế — nhiều người bị mất điện thoại là “mất luôn email” vì OTP gắn với SIM. Nhưng thực ra bạn vẫn lấy lại được, nếu trước đó có chuẩn bị hoặc biết cách."

# --- 4. Inference ---
print(f"Dang tao giong noi cho: {text}")
sentences = sent_tokenize(text)
wav_chunks = []
for sentence in sentences:
    if not sentence.strip():
        continue
    wav_chunk = model.inference(
        text=sentence,
        language="vi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
        enable_text_splitting=True,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

# --- 5. Luu output ---
out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
torchaudio.save("output.wav", out_wav, 24000)
print("Da luu ket qua vao output.wav")
