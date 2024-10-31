# Doc2Podcast

Convert documents into engaging podcasts using AI.

## Features

- Convert PDF documents into natural-sounding podcasts
- AI-powered content analysis and script generation using Claude-3 and CrewAI
- High-quality text-to-speech using F5-TTS and GLM-4-Voice
- Interactive Streamlit interface with step-by-step wizard
- Efficient document processing with semantic chunking
- Podcast library management
- Voice profile customization
- Progress tracking and error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doc2podcast.git
cd doc2podcast
```

2. Install dependencies:
```bash
pip install -e .
```

3. Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Follow the wizard steps:
   - Upload your PDF document
   - Configure script generation settings
   - Customize voice profiles
   - Generate your podcast

3. Access your podcasts in the Library section

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black src/ tests/
isort src/ tests/
```

## Current Status

### Implemented Features
- PDF document processing with semantic chunking
- AI-powered script generation using Claude-3 and CrewAI
- Basic voice synthesis with F5-TTS and GLM-4-Voice
- Interactive Streamlit interface
- Podcast library management
- Progress tracking and error handling

### In Progress
- Enhanced voice profile system
- Audio post-processing and enhancement
- Advanced CrewAI agent coordination
- Performance optimization

## License

MIT License - see LICENSE file for details.
