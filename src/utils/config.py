"""
Configuration management
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class Config:
    """Global configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path
        Example: config.get('tier1.thresholds.unsupported_similarity')
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self._config[key]
    
    def __repr__(self) -> str:
        return f"Config(loaded from {self.config_path})"


# Global config instance
config = Config()

