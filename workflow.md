flowchart TB
    subgraph Input
        A[Document] --> B[Document Processing Agent]
        B --> C[Text Cleaning]
        C --> D[Content Chunking]
    end
    
    subgraph Content Generation
        D --> E[Script Writer Agent]
        E --> F[Quality Control Agent]
        F --> G[Final Script]
    end
    
    subgraph Voice Synthesis
        G --> H[Voice Synthesis Agent]
        H --> I[Audio Post-processing]
        I --> J[Final Podcast]
    end
    
    %% Agent Details
    note1[Document Processing Agent:
    - PDF/Text extraction
    - Structure analysis
    - Content organization]
    B --> note1
    
    note2[Script Writer Agent:
    - Claude-3 integration
    - GLM-4 fallback
    - Dynamic prompting]
    E --> note2
    
    note3[Voice Synthesis Agent:
    - F5-TTS primary
    - GLM-4-Voice secondary
    - Voice profile management]
    H --> note3

    %% Process Flow
    note4[Processing Flow:
    1. Document intake & analysis
    2. Content structuring
    3. Script generation
    4. Quality verification
    5. Voice synthesis
    6. Audio enhancement]
    A --> note4

    %% Configuration
    note5[System Configuration:
    - YAML-based settings
    - Voice profiles
    - Agent behaviors
    - Task definitions]
    note4 --> note5

    %% CrewAI Integration
    note6[CrewAI Orchestration:
    - Agent collaboration
    - Task delegation
    - Quality control
    - Process monitoring]
    note5 --> note6
```

Key Components:

1. Document Processing Agent
- Handles document intake
- Extracts and cleans text
- Analyzes document structure
- Manages content chunks

2. Script Writer Agent
- Generates conversational scripts
- Uses Claude-3 for primary generation
- Implements GLM-4 as fallback
- Follows voice profile guidelines

3. Quality Control Agent
- Verifies content accuracy
- Checks conversation flow
- Ensures voice compatibility
- Suggests improvements

4. Voice Synthesis Agent
- Manages TTS processing
- Handles voice profile selection
- Coordinates audio generation
- Performs post-processing

System Integration:

1. CrewAI Framework
- Agent coordination
- Task delegation
- Process monitoring
- Quality assurance

2. Configuration Management
- YAML-based settings
- Voice profile definitions
- Agent behavior control
- Task specifications

3. Processing Pipeline
- Sequential task execution
- Error handling
- Progress tracking
- Result validation

4. Model Integration
- Claude-3 API
- GLM-4 API
- F5-TTS System
- Audio Processing

This workflow provides:
1. Modular processing
2. Quality control
3. Flexible configuration
4. Robust error handling
5. Progress monitoring
6. Result validation
