# Voice Clone - Vietnamese AI Voice Cloning

Clone giong noi tieng Viet bang AI, chay local tren Windows.

## Cong nghe

- **Model**: [viXTTS](https://huggingface.co/capleaf/viXTTS) - fine-tuned tu Coqui XTTS v2 cho tieng Viet
- **Framework**: [Coqui TTS (fork)](https://github.com/thinhlpg/TTS/tree/add-vietnamese-xtts) - ban mo rong ho tro Vietnamese tokenizer
- **Ngon ngu ho tro**: Tieng Viet (vi) + 17 ngon ngu khac (en, es, fr, de, ...)
- **Voice cloning**: Chi can ~6 giay audio mau

## Yeu cau he thong

- Windows 10/11
- Python 3.10
- RAM: toi thieu 8GB (khuyen nghi 16GB)
- GPU: khong bat buoc (co CUDA se nhanh hon)

## Cai dat

```bash
# 1. Tao va kich hoat moi truong ao
py -3.10 -m venv xtts-env
xtts-env\Scripts\activate

# 2. Cai TTS fork (ho tro tieng Viet)
pip install git+https://github.com/thinhlpg/TTS.git@add-vietnamese-xtts

# 3. Cai thu vien phu tro
pip install vinorm==2.0.7 underthesea==6.8.0 unidecode
pip install "torch>=2.1,<2.5" "torchaudio>=2.1,<2.5"
```

## Su dung

### Giao dien web (khuyen nghi)

```bash
python app.py
```

Mo trinh duyet: http://127.0.0.1:7860

Giao dien cho phep:
- Upload/thu am giong mau
- Nhap noi dung can doc
- Chon ngon ngu, toc do, creativity
- Nghe thu va tai ve ket qua

### Test nhanh (command line)

```bash
python test.py
```

## Cau truc project

```
voice-clone/
├── app.py           # Giao dien web (Gradio)
├── test.py          # Script clone giong (CLI)
├── voice.m4a        # File audio mau (giong cua ban)
├── model/           # Model viXTTS (tu dong tai)
├── output.wav       # Ket qua
├── xtts-env/        # Python virtual environment
└── README.md
```

## Luu y

- Cau duoi 10 tu tieng Viet co the chat luong kem
- Audio mau nen ro rang, khong nhieu
- Lan dau chay se tai model (~2GB), can internet
