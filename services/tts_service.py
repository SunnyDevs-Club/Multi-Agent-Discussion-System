"""Text-to-Speech Service Module
This module provides functionality to convert text input into speech audio using a pre-trained TTS model.
"""

import torch
import numpy as np
import soundfile as sf
import io
from TTS.api import TTS

import base64
import re

from services.storage_service import Agent

DEVICE: torch.device = None
if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

tts_model: TTS = None


def init_tts_model():
    global tts_model
    if tts_model is None:
        print(f"Loading XTTS-v2 model on {DEVICE}...")
        try:
            tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device=DEVICE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def basic_clean_text(text: str) -> str:   
    # 1. Remove thought/debug blocks first (if present)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. Normalize and remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Handle common LLM ending artifacts (example)
    if text.endswith("Thank you for reading."):
        text = text.removesuffix("Thank you for reading.")

    return text.strip()


def generate_audio_base64(text: str, agent: Agent, language: str = 'en') -> str:
    """Generate speech audio from text and return it as a base64-encoded string.

    Args:
        text (str): The input text to convert to speech.
        speaker_id (str): The identifier for the speaker voice.
        language (str): The language code for the TTS model.

    Returns:
        str: Base64-encoded string of the generated audio in WAV format.
    """
    if tts_model is None:
        raise Exception("TTS model not initialized. Call init_tts_model() first.")

    wav_files = agent.get_wav_files()
    wav = tts_model.tts(
        text,
        speaker_wav=wav_files,
        language=language,
        split_sentences=False,
        speed=1.0 if agent.agent_id != 'HONG' else 0.8,
        speaker=agent.agent_id
    )
    output_path = f"generated_{agent.agent_id}.wav"
    sf.write(output_path, np.array(wav), samplerate=tts_model.synthesizer.output_sample_rate, format='WAV')
    print(f"Audio saved locally at: {output_path}")

    # Convert numpy array to bytes
    byte_io = io.BytesIO()
    sf.write(byte_io, np.array(wav), samplerate=tts_model.synthesizer.output_sample_rate, format='WAV')
    byte_io.seek(0)

    # Encode to base64
    audio_base64 = base64.b64encode(byte_io.read()).decode('utf-8')

    return audio_base64

init_tts_model()
