# Hướng dẫn A → Z: Tự host AI clone giọng (Coqui XTTS v2) trên Windows

## 1. Giới thiệu công nghệ

-   Sử dụng Coqui TTS (XTTS v2)
-   Input: Text + audio mẫu
-   Output: Giọng nói giống bạn

Pipeline: Text → Model XTTS → Voice embedding → Audio

------------------------------------------------------------------------

## 2. Yêu cầu hệ thống

-   Windows 10/11
-   RAM: tối thiểu 8GB (khuyến nghị 16GB)
-   GPU (không bắt buộc)

------------------------------------------------------------------------

## 3. Setup môi trường

### Bước 1: Cài Python 3.10

Download: https://www.python.org/downloads/release/python-31011/ ✔ Tick
"Add Python to PATH"

### Bước 2: Tạo môi trường

    py -3.10 -m venv xtts-env
    xtts-env\Scripts\activate

### Bước 3: Cài thư viện

    pip install --upgrade pip
    pip install TTS==0.21.3

------------------------------------------------------------------------

## 4. Test clone giọng

Tạo file test.py:

    from TTS.api import TTS

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    tts.tts_to_file(
        text="Xin chào, đây là giọng nói của tôi",
        speaker_wav="voice.wav",
        language="vi",
        file_path="output.wav"
    )

Chạy:

    python test.py

------------------------------------------------------------------------

## 5. Tạo UI với Gradio

Tạo file app.py:

    import gradio as gr
    from TTS.api import TTS

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def generate_voice(text, audio, lang):
        output_path = "output.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=audio,
            language=lang,
            file_path=output_path
        )
        return output_path

    app = gr.Interface(
        fn=generate_voice,
        inputs=[
            gr.Textbox(label="Nhập nội dung"),
            gr.Audio(type="filepath", label="Upload giọng bạn"),
            gr.Dropdown(["vi", "en", "zh"], value="vi", label="Ngôn ngữ")
        ],
        outputs=gr.Audio(label="Kết quả"),
        title="Clone giọng XTTS v2"
    )

    app.launch()

Chạy:

    python app.py

Mở trình duyệt: http://127.0.0.1:7860

------------------------------------------------------------------------

## 6. Tối ưu chất lượng giọng

-   Thu 15--30s audio
-   Không noise
-   Giọng rõ

------------------------------------------------------------------------

## 7. Nâng cấp

-   Lưu giọng mặc định
-   Batch generate
-   Pipeline: text → voice → video

------------------------------------------------------------------------

## 8. Lỗi thường gặp

-   Sai Python version → dùng 3.10
-   Torch lỗi → cài đúng CUDA
-   Giọng không giống → audio kém

------------------------------------------------------------------------

## Kết luận

Bạn đã có hệ thống clone giọng: - Free 100% - Chạy local - Không phụ
thuộc nền tảng
