"""Configuration module for prompts and templates"""
from typing import Dict, Any
from .settings import Settings

class PromptManager:
    """Manages prompts and templates for the podcast generation system"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.agents_config = self.settings.agents_config
        self.tasks_config = self.settings.tasks_config
        self.speakers_config = self.settings.speakers_config
        
    def get_agent_prompt(self, agent_name: str) -> Dict[str, Any]:
        """Get agent configuration and prompt"""
        agent = self.agents_config["agents"].get(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found in configuration")
            
        return {
            "name": agent["name"],
            "role": agent["role"],
            "goal": agent["goal"],
            "backstory": agent["backstory"],
            "tools": agent["tools"],
            "allow_delegation": agent.get("allow_delegation", False),
            "verbose": agent.get("verbose", True)
        }
        
    def get_task_prompt(self, task_name: str) -> Dict[str, Any]:
        """Get task configuration and prompt"""
        task = self.tasks_config["tasks"].get(task_name)
        if not task:
            raise ValueError(f"Task {task_name} not found in configuration")
            
        return {
            "description": task["description"],
            "expected_output": task["expected_output"],
            "agent": task["agent"],
            "parameters": task["parameters"],
            "validation": task["validation"]
        }
        
    def get_voice_profile(self, category: str, name: str, profile_type: str) -> Dict[str, Any]:
        """Get voice profile configuration"""
        voices = self.speakers_config["voices"]
        if category not in voices:
            raise ValueError(f"Voice category {category} not found")
            
        speaker = voices[category].get(name)
        if not speaker:
            raise ValueError(f"Speaker {name} not found in {category}")
            
        profile = speaker["voice_profiles"].get(profile_type)
        if not profile:
            raise ValueError(f"Profile {profile_type} not found for {name}")
            
        return {
            "name": speaker["name"],
            "character_background": speaker["character_background"],
            "voice_profile": profile
        }
        
    def get_interview_prompt(self, format_type: str = "technical_deep_dive") -> Dict[str, Any]:
        """Get interview prompt configuration"""
        prompts = self.speakers_config["prompts"]
        if "interview" not in prompts:
            raise ValueError("Interview prompts not found in configuration")
            
        prompt = prompts["interview"].get(format_type)
        if not prompt:
            raise ValueError(f"Interview format {format_type} not found")
            
        return prompt
        
    def format_system_prompt(self, agent_name: str, task_name: str) -> str:
        """Format system prompt for an agent and task"""
        agent = self.get_agent_prompt(agent_name)
        task = self.get_task_prompt(task_name)
        
        return f"""You are {agent['name']}, {agent['role']}.
Your goal is to {agent['goal']}.
Background: {agent['backstory']}

Your current task:
{task['description']}

Expected output:
{task['expected_output']}

Please follow these validation requirements:
Required fields: {', '.join(task['validation']['required_fields'])}
Output format: {task['validation']['output_format']}
"""
        
    def format_interview_prompt(self, format_type: str = "technical_deep_dive") -> str:
        """Format interview prompt template"""
        prompt = self.get_interview_prompt(format_type)
        
        return f"""You are a professional podcast scriptwriter.

Format: {prompt['name']}
Description: {prompt['description']}

Roles:
Host ({prompt['roles']['host']['voice']}):
- Objective: {prompt['roles']['host']['objective']}
- Style: {prompt['roles']['host']['style']}

Guest ({prompt['roles']['guest']['voice']}):
- Objective: {prompt['roles']['guest']['objective']}
- Style: {prompt['roles']['guest']['style']}

Structure:
1. Introduction:
{prompt['structure']['introduction']['template']}

2. Main Discussion:
Segments: {', '.join(prompt['structure']['main_discussion']['segments'])}

3. Conclusion:
{prompt['structure']['conclusion']['template']}
"""
