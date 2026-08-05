# tools/tts_coqui.py
import os
import tempfile

def synthesize_text_to_wav(text: str, out_path: str = None) -> str:
    out_path = out_path or tempfile.mktemp(suffix=".mp3")
    
    # Primary attempt: gTTS (Google Text-to-Speech)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(out_path)
        return out_path
    except Exception as e:
        print(f"gTTS error: {e}, attempting pyttsx3 fallback...")
    
    # Secondary attempt: pyttsx3 (Offline Windows TTS)
    try:
        import pyttsx3
        out_path = out_path.replace(".mp3", ".wav")
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    except Exception as e:
        print(f"pyttsx3 error: {e}")
        
    return out_path
