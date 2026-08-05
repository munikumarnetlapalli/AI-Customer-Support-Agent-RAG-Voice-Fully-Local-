# tools/stt_whisper.py
import tempfile
import os

_model = None

def get_whisper_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model("tiny")  # local tiny model
    return _model

def transcribe_audio_bytes(audio_bytes: bytes, format: str = "wav") -> str:
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tf:
        tf.write(audio_bytes)
        tf.flush()
        tmp_path = tf.name
    
    try:
        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        text = result.get("text", "")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return text
