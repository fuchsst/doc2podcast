from typing import Dict, Any, Optional, Tuple
import random
import sys
from pathlib import Path
import torch
import logging

from ..config.settings import Settings
from ..utils.cache_manager import cache_manager
from .tts import (
    DiT,
    UNetT,
    seed_everything,
    load_audio,
    save_audio,
    chunk_text,
    remove_silence,
    normalize_audio,
    cross_fade
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F5TTSClient:
    MODEL_URLS = {
        "F5-TTS": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
        "E2-TTS": "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
    }
    
    def __init__(
        self,
        settings: Settings,
        model_type: str = "F5-TTS",
        ckpt_file: str = "",
        vocab_file: str = "",
        ode_method: str = "euler",
        use_ema: bool = True,
        local_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize F5TTS client
        
        Args:
            model_type: Type of model to use ("F5-TTS" or "E2-TTS")
            ckpt_file: Path to model checkpoint file
            vocab_file: Path to vocabulary file
            ode_method: ODE solver method
            use_ema: Whether to use EMA weights
            local_path: Path to local model files
            device: Device to use for inference
            cache_dir: Directory to cache downloaded models
        """
        self.settings = settings
        
        # Initialize parameters
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1
        
        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
        # Initialize model
        self.model_type = model_type
        self.initialize_model(
            model_type=model_type,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ode_method=ode_method,
            use_ema=use_ema,
            local_path=local_path
        )
        
    def initialize_model(
        self,
        model_type: str,
        ckpt_file: str,
        vocab_file: str,
        ode_method: str,
        use_ema: bool,
        local_path: Optional[str]
    ):
        """Initialize TTS model"""
        logger.info(f"Initializing {model_type} model...")
        
        # Configure model
        if model_type == "F5-TTS":
            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4
            )
            model_cls = DiT
        elif model_type == "E2-TTS":
            model_cfg = dict(
                dim=1024,
                depth=24,
                heads=16,
                ff_mult=4
            )
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Create model
        self.model = model_cls(**model_cfg).to(self.device)
        
        # Load checkpoint if provided
        if ckpt_file:
            logger.info(f"Loading checkpoint from {ckpt_file}")
            state_dict = torch.load(ckpt_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        logger.info("Model initialized successfully")
        
    @cache_manager.cache()
    def generate_audio(
        self,
        text: str,
        speaker: str,
        voice_config: Dict[str, Any],
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_path: Optional[Path] = None,
        seed: int = -1,
        remove_silences: bool = False,
        cross_fade_duration: float = 0.15,
        speed: float = 1.0,
        show_progress: bool = True
    ) -> Tuple[bytes, float]:
        """
        Generate audio using F5-TTS
        
        Args:
            text: Text to synthesize
            speaker: Speaker identifier
            voice_config: Voice configuration parameters
            ref_audio: Optional reference audio file path
            ref_text: Optional reference text
            output_path: Optional path to save the audio file
            seed: Random seed for reproducible generation (-1 for random)
            remove_silences: Whether to remove silence from the generated audio
            cross_fade_duration: Duration of cross-fade between segments
            speed: Speed/pace adjustment factor
            show_progress: Whether to show progress bar during generation
            
        Returns:
            Tuple containing:
            - Audio data as bytes
            - Sample rate of the audio
        """
        try:
            # Set random seed
            if seed == -1:
                seed = random.randint(0, sys.maxsize)
            self.seed = seed
            seed_everything(seed)
            
            # Load reference audio if provided
            if ref_audio:
                ref_audio_tensor, _ = load_audio(
                    ref_audio,
                    target_sample_rate=self.target_sample_rate
                )
            else:
                ref_audio_tensor = None
                
            # Extract voice parameters
            voice_params = voice_config.get("voice_parameters", {})
            pace = voice_params.get("pace", 1.0) * speed
            
            # Split text into chunks
            text_chunks = chunk_text(text)
            
            # Generate audio for each chunk
            audio_chunks = []
            for chunk in text_chunks:
                # Generate audio
                with torch.inference_mode():
                    audio = self.model(
                        text=chunk,
                        ref_audio=ref_audio_tensor,
                        ref_text=ref_text,
                        speed=pace
                    )
                    
                # Post-process audio
                if remove_silences:
                    audio = remove_silence(audio)
                    
                audio = normalize_audio(audio, target_level=self.target_rms)
                audio_chunks.append(audio)
                
            # Combine chunks with crossfade
            final_audio = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                final_audio = cross_fade(
                    final_audio,
                    chunk,
                    fade_duration=cross_fade_duration,
                    sample_rate=self.target_sample_rate
                )
                
            # Save to file if output path provided
            if output_path:
                save_audio(
                    final_audio,
                    str(output_path),
                    sample_rate=self.target_sample_rate
                )
                
            return final_audio.numpy().tobytes(), self.target_sample_rate
                
        except Exception as e:
            raise Exception(f"F5-TTS generation failed: {str(e)}")
            
    def get_voice_config(self, speaker_type: str = "professional", style: str = "technical") -> Dict[str, Any]:
        """
        Get voice configuration from speakers.yaml
        
        Args:
            speaker_type: Type of speaker ("professional" or "casual")
            style: Voice style/profile to use (e.g. "technical", "enthusiastic")
            
        Returns:
            Dictionary containing voice configuration parameters
        """
        speakers = self.settings.speakers_config["voices"]
        
        # Find matching speaker configuration
        if speaker_type not in speakers:
            speaker_type = next(iter(speakers))
        speaker_config = next(iter(speakers[speaker_type].values()))
            
        # Get voice profile for requested style
        voice_profile = speaker_config["voice_profiles"].get(style)
        if not voice_profile:
            # Fallback to first available profile
            style = next(iter(speaker_config["voice_profiles"]))
            voice_profile = speaker_config["voice_profiles"][style]
            
        return {
            "name": speaker_config["name"],
            "model": voice_profile.get("model", "F5-TTS"),
            "reference_audio": voice_profile.get("reference_audio"),
            "reference_text": voice_profile.get("reference_text"),
            "style_tags": voice_profile.get("style_tags", []),
            "voice_parameters": voice_profile.get("voice_parameters", {
                "pace": 1.0,
                "pitch": 0.5,
                "energy": 0.7,
                "emotion": "professional",
                "variation": 0.1
            })
        }
            
    def get_last_seed(self) -> int:
        """Get the seed used in the last generation"""
        return self.seed
