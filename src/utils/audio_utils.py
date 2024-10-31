from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess
import tempfile
import shutil
import os

class AudioProcessor:
    def __init__(self):
        # Try multiple common FFmpeg locations
        ffmpeg_locations = [
            'ffmpeg',  # System PATH
            r'C:\ffmpeg\bin\ffmpeg.exe',  # Common Windows install location
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        ]
        
        for location in ffmpeg_locations:
            if os.path.isfile(location) or shutil.which(location):
                self.ffmpeg_path = location
                break
        else:
            raise RuntimeError("ffmpeg not found. Please ensure FFmpeg is installed and in system PATH")
            
    def combine_audio_segments(
        self,
        segment_paths: List[Path],
        output_path: Path,
        format: str = "mp3",
        bitrate: str = "192k",
        parameters: Optional[List[str]] = None
    ) -> Path:
        """Combine multiple audio segments into single file"""
        try:
            # Verify all input files exist
            for path in segment_paths:
                if not path.exists():
                    raise FileNotFoundError(f"Audio segment not found: {path}")
                    
            # Create temporary file list
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                for path in segment_paths:
                    # Use forward slashes for FFmpeg compatibility
                    f.write(f"file '{str(path.absolute()).replace(os.sep, '/')}'\n")
                file_list = Path(f.name)
                
            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list)
            ]
            
            # Add format-specific encoding parameters
            if format.lower() == "mp3":
                cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate])
            elif format.lower() == "wav":
                cmd.extend(["-c:a", "pcm_s16le"])
            
            # Add additional parameters
            if parameters:
                cmd.extend(parameters)
                
            # Add output path
            cmd.extend(["-y", str(output_path)])
            
            # Execute command
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Cleanup
            file_list.unlink()
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Audio combination error: {str(e)}")
            
    def normalize_audio(
        self,
        input_path: Path,
        target_level: float = -23.0,
        output_path: Optional[Path] = None
    ) -> Path:
        """Normalize audio levels"""
        try:
            output_path = output_path or input_path.with_suffix(f".norm{input_path.suffix}")
            
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
                "-y", str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Audio normalization error: {str(e)}")
            
    def remove_silence(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        silence_threshold: float = -50.0,
        min_silence_duration: float = 0.5
    ) -> Path:
        """Remove silence from audio"""
        try:
            output_path = output_path or input_path.with_suffix(f".trimmed{input_path.suffix}")
            
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-af", f"silenceremove=stop_periods=-1:stop_duration={min_silence_duration}:stop_threshold={silence_threshold}dB",
                "-y", str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Silence removal error: {str(e)}")

    def adjust_speed(
        self,
        input_path: Path,
        speed: float = 1.0,
        output_path: Optional[Path] = None
    ) -> Path:
        """Adjust audio playback speed"""
        try:
            output_path = output_path or input_path.with_suffix(f".speed{input_path.suffix}")
            
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-filter:a", f"atempo={speed}",
                "-y", str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Speed adjustment error: {str(e)}")

    def add_silence(
        self,
        input_path: Path,
        duration: float,
        position: str = "end",
        output_path: Optional[Path] = None
    ) -> Path:
        """Add silence to audio file
        
        Args:
            input_path: Path to input audio file
            duration: Duration of silence in seconds
            position: Where to add silence ("start", "end", or "both")
            output_path: Optional output path
        """
        try:
            output_path = output_path or input_path.with_suffix(f".silence{input_path.suffix}")
            
            # Create silence file
            silence_file = Path(tempfile.mktemp(suffix=".wav"))
            silence_cmd = [
                self.ffmpeg_path,
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
                "-y", str(silence_file)
            ]
            subprocess.run(silence_cmd, check=True, capture_output=True, text=True)
            
            # Create file list based on position
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                if position in ["start", "both"]:
                    f.write(f"file '{str(silence_file.absolute()).replace(os.sep, '/')}'\n")
                f.write(f"file '{str(input_path.absolute()).replace(os.sep, '/')}'\n")
                if position in ["end", "both"]:
                    f.write(f"file '{str(silence_file.absolute()).replace(os.sep, '/')}'\n")
                file_list = Path(f.name)
            
            # Combine files
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-y", str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Cleanup
            silence_file.unlink()
            file_list.unlink()
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error adding silence: {str(e)}")

    def save_audio(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "mp3",
        bitrate: str = "192k",
        parameters: Optional[List[str]] = None
    ) -> Path:
        """Save audio file with specified format and parameters"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path)
            ]
            
            # Add format-specific encoding parameters
            if format.lower() == "mp3":
                cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate])
            elif format.lower() == "wav":
                cmd.extend(["-c:a", "pcm_s16le"])
                
            if parameters:
                cmd.extend(parameters)
                
            cmd.extend(["-y", str(output_path)])
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {str(e)}")

    def load_audio(
        self,
        input_path: Path,
        output_format: str = "wav",
        parameters: Optional[List[str]] = None
    ) -> Path:
        """Load and convert audio file to specified format"""
        try:
            output_path = input_path.with_suffix(f".{output_format}")
            
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path)
            ]
            
            # Add format-specific encoding parameters
            if output_format.lower() == "mp3":
                cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
            elif output_format.lower() == "wav":
                cmd.extend(["-c:a", "pcm_s16le"])
                
            if parameters:
                cmd.extend(parameters)
                
            cmd.extend(["-y", str(output_path)])
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {str(e)}")

audio_utils = AudioProcessor()

# Export functions to match __init__.py interface
def combine_audio_segments(*args, **kwargs):
    return audio_utils.combine_audio_segments(*args, **kwargs)

def adjust_speed(*args, **kwargs):
    return audio_utils.adjust_speed(*args, **kwargs)

def normalize_audio(*args, **kwargs):
    return audio_utils.normalize_audio(*args, **kwargs)

def add_silence(*args, **kwargs):
    return audio_utils.add_silence(*args, **kwargs)

def remove_silence(*args, **kwargs):
    return audio_utils.remove_silence(*args, **kwargs)

def save_audio(*args, **kwargs):
    return audio_utils.save_audio(*args, **kwargs)

def load_audio(*args, **kwargs):
    return audio_utils.load_audio(*args, **kwargs)

# Add merge_audio_files as an alias for combine_audio_segments
def merge_audio_files(audio_paths: List[str], output_path: str) -> str:
    """
    Merge multiple audio files into a single file.
    This is an alias for combine_audio_segments to maintain compatibility.
    
    Args:
        audio_paths (List[str]): List of paths to audio files to merge
        output_path (str): Path where the merged audio file will be saved
        
    Returns:
        str: Path to the merged audio file
    """
    # Convert string paths to Path objects
    paths = [Path(p) for p in audio_paths]
    output = Path(output_path)
    
    # Use existing combine_audio_segments function
    result = audio_utils.combine_audio_segments(paths, output)
    
    # Return string path to match expected interface
    return str(result)
