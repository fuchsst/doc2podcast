doc2podcast/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example                # Template for API keys and configs
├── config/                     # Configuration files
│   ├── agents.yaml            # CrewAI agent definitions
│   ├── project.yaml           # Project-wide settings
│   ├── speakers.yaml          # Voice profiles and prompts
│   ├── tasks.yaml             # CrewAI task definitions
│   └── schemas/               # JSON schemas for config validation
├── src/
│   ├── __init__.py
│   ├── app.py                 # Main application class
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py        # Settings management
│   │   └── prompts.py         # Claude prompt templates
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Document handling
│   │   ├── text_cleaner.py       # Text preprocessing
│   │   └── chunk_manager.py      # Content chunking
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── claude_client.py      # Claude API integration
│   │   ├── glm4_client.py        # GLM-4 API integration
│   │   ├── f5tts_client.py       # F5-TTS integration
│   │   ├── script_generator.py   # Script generation
│   │   ├── voice_generator.py    # Voice synthesis
│   │   └── tts/                  # TTS utilities
│   │       ├── __init__.py
│   │       ├── models.py         # TTS model interfaces
│   │       └── utils.py          # TTS helper functions
│   ├── models/
│   │   ├── __init__.py
│   │   └── podcast_script.py     # Data models
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── podcast_pipeline.py   # CrewAI workflow
│   └── utils/
│       ├── __init__.py
│       ├── text_utils.py         # Text processing utilities
│       ├── audio_utils.py        # Audio processing utilities
│       └── cache_manager.py      # Caching system
├── streamlit_app/               # Streamlit interface
│   ├── __init__.py
│   ├── app.py                   # Main Streamlit app
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── home.py             # Upload interface
│   │   ├── processing.py       # Processing status
│   │   └── library.py          # Generated podcasts
│   └── components/
│       ├── __init__.py
│       ├── file_uploader.py    # File upload component
│       ├── status_tracker.py   # Progress tracking
│       └── audio_player.py     # Audio playback
├── models/                     # Downloaded model cache
└── cache/                      # Processing cache

Key Implementation Details:

1. CrewAI Integration:
- Agents defined in config/agents.yaml
- Tasks defined in config/tasks.yaml
- Pipeline orchestration in src/pipeline/podcast_pipeline.py

2. Voice Synthesis:
- Primary: F5-TTS integration
- Secondary: GLM-4-Voice support
- Voice profiles in config/speakers.yaml

3. Content Generation:
- Claude-3 for script generation
- GLM-4 as fallback option
- Prompt templates in src/config/prompts.py

4. Configuration:
- YAML-based configuration
- JSON schema validation
- Environment variables via .env

5. User Interface:
- Streamlit-based web interface
- Real-time processing status
- Audio preview and download

6. Processing Pipeline:
- Document analysis
- Content extraction
- Script generation
- Voice synthesis
- Audio post-processing

This structure provides:
1. Clear separation of concerns
2. Modular components
3. Flexible configuration
4. Multiple model support
5. Caching capabilities
6. Comprehensive logging
