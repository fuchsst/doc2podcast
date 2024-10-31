import os
from pathlib import Path

# Set HF_HOME to ./models directory before any imports
models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(models_dir)

from .app import DocToPodcast
from .config.settings import Settings
from .models.podcast_script import PodcastScript

__version__ = "0.1.0"
__all__ = ["DocToPodcast", "Settings", "PodcastScript"]
