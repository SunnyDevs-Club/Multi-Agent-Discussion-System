import json
from pathlib import Path
from utils import get_agent_system_prompt

AGENT_DATA_PATH = Path(__file__).parent.parent / "agents_data"
PROMPT_DIR = AGENT_DATA_PATH / "sys_prompts"
WAV_DIR = AGENT_DATA_PATH / "speaker_wavs"


class Agent:
    def __init__(self, agent_id: str, model_name: str):
        self.agent_id: str = agent_id 
        self.model_name: str = model_name
        self.wav_file_path: Path = WAV_DIR / f"{agent_id.upper()}.wav"
        self.system_prompt_file: Path = PROMPT_DIR / f"{agent_id.lower()}.yaml"
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
        }
    
    def get_system_prompt(self) -> str:
        return get_agent_system_prompt(self.system_prompt_file)
    
    def get_wav_files(self) -> list[str | Path]:
        wav_list = [str(p) for p in (WAV_DIR / self.agent_id).glob('*.wav')]
        return wav_list

    def __repr__(self):
        return f"Agent(agent_id={self.agent_id}, model_name={self.model_name})"


class AgentStorage:
    def __init__(self, filepath: Path | str = AGENT_DATA_PATH / "agents.json"):
        self.filepath: Path = Path(filepath) if isinstance(filepath, str) else filepath 
        self.agents: dict[str, Agent] = self._load_agents()

    def _load_agents(self) -> dict[str, Agent]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                return {
                    agent_id: Agent(agent_id, model_name)
                    for agent_id, model_name in data.items()
                }
        except FileNotFoundError:
            print(f"Agent file not found: {self.filepath}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.filepath}: {e}")
            return {}

    def get_agent(self, agent_id: str) -> Agent | None:
        return self.agents.get(agent_id, None)
    
    def get_all_agents(self) -> dict[str, Agent]:
        return self.agents
    
    def add_agent(self, agent_id: str, model_name: str):
        self.agents[agent_id] = Agent(agent_id, model_name)
        self._save_agents()
    
    def _save_agents(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump({agent_id: agent.to_dict() for agent_id, agent in self.agents.items()}, f, indent=4)
        except Exception as e:
            print(f"Error saving agents to {self.filepath}: {e}")
            raise e

    def remove_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._save_agents()
        else:
            raise KeyError(f"Agent with agent_id '{agent_id}' does not exist.")
    
    def update_agent(self, agent_id: str, model_name: str = None):
        if agent_id in self.agents:
            if model_name:
                self.agents[agent_id]["model_name"] = model_name
            self._save_agents()
        else:
            raise KeyError(f"Agent with agent_id '{agent_id}' does not exist.")
