from typing import Dict, Any, Optional, Tuple
import random
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.generation.streamers import BaseStreamer
from queue import Queue
from threading import Thread

from ..config.settings import Settings
from ..utils.cache_manager import cache_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenStreamer(BaseStreamer):
    """Custom token streamer for GLM-4-Voice generation"""
    def __init__(self, skip_prompt: bool = False, timeout: Optional[float] = None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        return value

class GLM4Client:
    MODEL_PATHS = {
        "model": "THUDM/glm-4-voice-9b",
        "tokenizer": "THUDM/glm-4-voice-tokenizer",
        "decoder": "THUDM/glm-4-voice-decoder"
    }
    
    def __init__(
        self,
        settings: Settings,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize GLM4 client
        
        Args:
            model_path: Path to GLM-4-Voice model
            tokenizer_path: Path to tokenizer model
            decoder_path: Path to decoder model
            dtype: Model dtype ("bfloat16" or "int4")
            device: Device to use for inference
            cache_dir: Directory to cache downloaded models
        """
        self.settings = settings
        
        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
        # Initialize parameters
        self.target_sample_rate = 24000
        self.seed = -1
        self.dtype = dtype
        self.cache_dir = cache_dir
        
        # Initialize models
        self.initialize_models(
            model_path or self.MODEL_PATHS["model"],
            tokenizer_path or self.MODEL_PATHS["tokenizer"],
            decoder_path or self.MODEL_PATHS["decoder"]
        )
        
        logger.info("GLM-4-Voice client initialized successfully")
        
    def initialize_models(
        self,
        model_path: str,
        tokenizer_path: str,
        decoder_path: str
    ):
        """Initialize GLM-4-Voice models"""
        try:
            # Configure quantization if using int4
            bnb_config = None
            if self.dtype == "int4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            # Load main model
            logger.info(f"Loading GLM-4-Voice model from {model_path}")
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map={"": 0},
                cache_dir=self.cache_dir
            ).eval()
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Load decoder
            logger.info(f"Loading decoder from {decoder_path}")
            self.decoder = AutoModel.from_pretrained(
                decoder_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).eval().to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
    @torch.inference_mode()
    def _generate_tokens(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
        timeout: Optional[float] = None
    ) -> list:
        """Generate audio tokens from text"""
        try:
            # Tokenize input
            inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Setup streamer
            streamer = TokenStreamer(skip_prompt=True, timeout=timeout)
            
            # Generate in separate thread
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    streamer=streamer
                )
            )
            thread.start()
            
            # Collect tokens
            tokens = []
            for token in streamer:
                tokens.append(token)
                
            return tokens
            
        except Exception as e:
            raise Exception(f"Token generation failed: {str(e)}")
        
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
        Generate audio using GLM-4-Voice
        
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
            torch.manual_seed(seed)
            
            # Extract voice parameters
            voice_params = voice_config.get("voice_parameters", {})
            
            # Prepare prompt with voice control
            prompt = self._prepare_voice_prompt(text, voice_params)
            
            # Generate audio tokens
            tokens = self._generate_tokens(
                prompt=prompt,
                temperature=voice_params.get("temperature", 0.9),
                top_p=voice_params.get("top_p", 0.9),
                max_new_tokens=256
            )
            
            # Convert tokens to audio using decoder
            audio_data = self._decode_audio_tokens(tokens)
            
            # Apply voice parameters
            audio_data = self._apply_voice_params(
                audio_data,
                speed=speed,
                voice_params=voice_params
            )
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                    
            return audio_data, self.target_sample_rate
                
        except Exception as e:
            raise Exception(f"GLM-4-Voice generation failed: {str(e)}")
            
    def _prepare_voice_prompt(self, text: str, voice_params: Dict[str, Any]) -> str:
        """Prepare prompt with voice control instructions"""
        # Extract voice parameters
        emotion = voice_params.get("emotion", "professional")
        dialect = voice_params.get("dialect", "")
        pace = voice_params.get("pace", 1.0)
        
        # Build voice control prompt
        voice_instructions = []
        if emotion:
            voice_instructions.append(f"Use a {emotion} voice")
        if dialect:
            voice_instructions.append(f"Speak in {dialect} dialect")
        if pace != 1.0:
            speed_desc = "faster" if pace > 1 else "slower"
            voice_instructions.append(f"Speak {speed_desc}")
            
        prompt = text
        if voice_instructions:
            prompt = f"{', '.join(voice_instructions)}. {text}"
            
        return prompt
        
    @torch.inference_mode()
    def _decode_audio_tokens(self, tokens: list) -> bytes:
        """Convert audio tokens to waveform using decoder"""
        try:
            # Convert tokens to tensor
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            # Generate audio with decoder
            audio = self.decoder.generate(tokens_tensor)
            
            # Convert to bytes
            return audio.cpu().numpy().tobytes()
            
        except Exception as e:
            raise Exception(f"Audio decoding failed: {str(e)}")
        
    def _apply_voice_params(
        self,
        audio_data: bytes,
        speed: float,
        voice_params: Dict[str, Any]
    ) -> bytes:
        """Apply voice parameter adjustments to audio"""
        try:
            # Convert bytes to numpy array
            audio = np.frombuffer(audio_data, dtype=np.float32)
            
            # Apply speed adjustment
            if speed != 1.0:
                # TODO: Implement proper speed adjustment
                pass
                
            # Apply other voice parameters
            # TODO: Implement other parameter adjustments
            
            return audio.tobytes()
            
        except Exception as e:
            raise Exception(f"Voice parameter application failed: {str(e)}")
            
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
            "model": voice_profile.get("model", "GLM-4-Voice"),
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
