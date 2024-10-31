"""Utility functions for TTS generation"""
import re
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_audio(
    audio_path: str,
    target_sample_rate: int = 24000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file"""
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        
    # Normalize if requested
    if normalize:
        audio = audio / audio.abs().max()
        
    return audio, target_sample_rate

def save_audio(
    audio: torch.Tensor,
    path: str,
    sample_rate: int = 24000
):
    """Save audio tensor to file"""
    torchaudio.save(path, audio, sample_rate)

def chunk_text(text: str, max_chars: int = 150) -> List[str]:
    """Split text into chunks for processing"""
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def remove_silence(
    audio: torch.Tensor,
    threshold: float = -40.0,
    min_silence_duration: float = 0.1,
    sample_rate: int = 24000
) -> torch.Tensor:
    """Remove silence from audio"""
    # Convert to numpy for processing
    audio_np = audio.numpy()
    
    # Calculate RMS energy
    frame_length = int(min_silence_duration * sample_rate)
    hop_length = frame_length // 4
    
    # Calculate energy in frames
    frames = np.array([
        np.sqrt(np.mean(frame**2))
        for frame in np.array_split(audio_np, len(audio_np) // hop_length)
    ])
    
    # Find non-silent frames
    threshold = 10**(threshold / 20)
    mask = frames > threshold
    
    # Reconstruct audio
    result = np.concatenate([
        audio_np[i*hop_length:(i+1)*hop_length]
        for i, m in enumerate(mask) if m
    ])
    
    return torch.from_numpy(result)

def normalize_audio(
    audio: torch.Tensor,
    target_level: float = -23.0
) -> torch.Tensor:
    """Normalize audio to target RMS level"""
    rms = torch.sqrt(torch.mean(audio**2))
    scalar = 10**(target_level/20) / (rms + 1e-10)
    return audio * scalar

def cross_fade(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    fade_duration: float = 0.1,
    sample_rate: int = 24000
) -> torch.Tensor:
    """Apply cross-fade between two audio segments"""
    fade_length = int(fade_duration * sample_rate)
    
    if fade_length >= len(audio1) or fade_length >= len(audio2):
        return torch.cat([audio1, audio2])
        
    # Create fade curves
    fade_in = torch.linspace(0, 1, fade_length)
    fade_out = torch.linspace(1, 0, fade_length)
    
    # Apply fades
    audio1_end = audio1[-fade_length:] * fade_out
    audio2_start = audio2[:fade_length] * fade_in
    
    # Combine with crossfade
    crossfade = audio1_end + audio2_start
    
    return torch.cat([
        audio1[:-fade_length],
        crossfade,
        audio2[fade_length:]
    ])
