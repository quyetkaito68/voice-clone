# Voice Clone - Clone giọng nói tiếng Việt bằng AI

Clone giọng nói tiếng Việt bằng AI, chạy hoàn toàn trên máy local (Windows).

## Công nghệ

- **Model**: [viXTTS](https://huggingface.co/capleaf/viXTTS) — fine-tuned từ Coqui XTTS v2 dành riêng cho tiếng Việt
- **Framework**: [Coqui TTS (fork)](https://github.com/thinhlpg/TTS/tree/add-vietnamese-xtts) — bản mở rộng hỗ trợ Vietnamese tokenizer
- **Giao diện**: [Gradio](https://gradio.app/) — giao diện web hiện đại, dễ sử dụng
- **Ngôn ngữ hỗ trợ**: Tiếng Việt (vi) + 17 ngôn ngữ khác (en, es, fr, de, ...)
- **Voice cloning**: Chỉ cần ~6 giây audio mẫu

## Yêu cầu hệ thống

- Windows 10/11
- Python 3.10
- RAM: tối thiểu 8GB (khuyến nghị 16GB)
- GPU NVIDIA (khuyến nghị) hoặc CPU

## Cài đặt

### Bước 1: Tạo môi trường ảo

```bash
py -3.10 -m venv xtts-env
xtts-env\Scripts\activate
```

### Bước 2: Cài TTS fork (hỗ trợ tiếng Việt)

```bash
pip install git+https://github.com/thinhlpg/TTS.git@add-vietnamese-xtts
```

### Bước 3: Cài thư viện phụ trợ

```bash
pip install underthesea==6.8.0 unidecode gradio pydub
```

### Bước 4: Cài PyTorch

**Nếu có GPU NVIDIA (khuyến nghị — nhanh hơn rất nhiều):**

```bash
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**Nếu chỉ có CPU:**

```bash
pip install torch==2.4.1 torchaudio==2.4.1
```

### Bước 5: Cài FFmpeg (cần cho xử lý audio)

```bash
winget install ffmpeg
```

## Sử dụng

### Giao diện web (khuyến nghị)

```bash
python app.py
```

Mở trình duyệt: http://127.0.0.1:7860

Giao diện cho phép:
- Upload hoặc thu âm giọng mẫu trực tiếp từ mic
- Nhập nội dung cần đọc
- Chọn ngôn ngữ, tốc độ, mức creativity
- Nghe thử và tải về kết quả

### Test nhanh (command line)

```bash
python test.py
```

## Cấu trúc project

```
voice-clone/
├── app.py           # Giao diện web (Gradio)
├── test.py          # Script clone giọng (CLI)
├── voice.wav        # File audio mẫu (giọng của bạn)
├── model/           # Model viXTTS (tự động tải)
├── output.wav       # Kết quả
├── xtts-env/        # Python virtual environment
└── README.md
```

## Lưu ý

- Câu dưới 10 từ tiếng Việt có thể cho chất lượng kém
- Audio mẫu nên rõ ràng, không nhiễu, dài 6-30 giây
- Lần đầu chạy sẽ tự động tải model (~2GB), cần internet
- Chạy trên GPU (RTX 2060 trở lên) sẽ nhanh hơn CPU rất nhiều (vài giây thay vì vài phút)
