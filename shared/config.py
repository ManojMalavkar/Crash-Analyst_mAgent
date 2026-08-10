"""Central Configuration Management for SafetyAgent Platform.

Provides a unified configuration system that loads settings from:
1. Default values (sensible defaults for all settings)
2. Environment variables (via .env file or system env)
3. Runtime overrides (programmatic configuration)

Usage:
    from shared.config import settings
    
    # Access configuration
    model = settings.llm.primary_model
    db_path = settings.paths.log_db
    
    # Override at runtime
    settings.llm.primary_model = "claude-sonnet-4-20250514"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


# =============================================================================
# Project Root Detection
# =============================================================================

def _find_project_root() -> Path:
    """Find the project root by looking for known marker files."""
    current = Path(__file__).resolve().parent.parent
    markers = ["requirements.txt", "README.md", ".gitignore"]
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current


PROJECT_ROOT = _find_project_root()


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class LLMConfig:
    """LLM model and endpoint configuration."""
    
    # Primary model (tried first)
    primary_model: str = field(
        default_factory=lambda: os.getenv("LLM_PRIMARY_MODEL", "claude-sonnet-4-20250514")
    )
    
    # Fallback models (tried in order if primary fails)
    fallback_models: list = field(
        default_factory=lambda: [
            os.getenv("LLM_FALLBACK_1", "llama-3-70b-instruct"),
            os.getenv("LLM_FALLBACK_2", "llama-3-8b-instruct"),
        ]
    )
    
    # API endpoint (Databricks AI Gateway or OpenAI-compatible)
    api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_BASE_URL",
            "https://your-workspace.databricks.net/serving-endpoints"
        )
    )
    
    # Authentication
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    
    # Generation parameters
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )
    
    # Retry configuration
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )
    retry_base_delay: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
    )
    retry_max_delay: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_MAX_DELAY", "60.0"))
    )
    
    # Timeout (seconds)
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    dimension: int = 384  # bge-small-en-v1.5 output dimension
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    )


@dataclass
class PathConfig:
    """File system paths for all agents."""
    
    # Project root
    root: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    # Shared paths
    log_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "logs"
    )
    log_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "logs" / "usage.db"
    )
    
    # Agent-specific paths
    ansa_knowledge_base: Path = field(
        default_factory=lambda: PROJECT_ROOT / "01_ANSA_ApiAgent" / "knowledge-base"
    )
    ansa_vector_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "01_ANSA_ApiAgent" / "vector_db"
    )
    
    pydyna_knowledge_base: Path = field(
        default_factory=lambda: PROJECT_ROOT / "02_PyDyna_Agent" / "knowledge-base"
    )
    pydyna_vector_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "02_PyDyna_Agent" / "vector_db"
    )
    
    modelcheck_checklists: Path = field(
        default_factory=lambda: PROJECT_ROOT / "03_ModelCheck_Agent" / "checklists"
    )
    modelcheck_reports: Path = field(
        default_factory=lambda: PROJECT_ROOT / "03_ModelCheck_Agent" / "reports"
    )
    
    hpc_templates: Path = field(
        default_factory=lambda: PROJECT_ROOT / "05_HPC_Orchestrator" / "templates"
    )
    hpc_pipelines: Path = field(
        default_factory=lambda: PROJECT_ROOT / "05_HPC_Orchestrator" / "pipelines"
    )
    
    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        dirs = [
            self.log_dir,
            self.ansa_knowledge_base,
            self.ansa_vector_db,
            self.pydyna_knowledge_base,
            self.pydyna_vector_db,
            self.modelcheck_checklists,
            self.modelcheck_reports,
            self.hpc_templates,
            self.hpc_pipelines,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    
    # Tool-calling loop limits
    max_tool_calls: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_CALLS", "10"))
    )
    
    # Context window management
    max_context_messages: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_CONTEXT_MESSAGES", "20"))
    )
    
    # RAG retrieval settings
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RETRIEVAL_TOP_K", "5"))
    )
    retrieval_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("AGENT_RETRIEVAL_THRESHOLD", "0.7"))
    )
    
    # Code generation settings
    code_review_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_CODE_REVIEW", "true").lower() == "true"
    )


@dataclass
class UIConfig:
    """Gradio UI configuration."""
    
    host: str = field(
        default_factory=lambda: os.getenv("UI_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("UI_PORT", "7860"))
    )
    share: bool = field(
        default_factory=lambda: os.getenv("UI_SHARE", "false").lower() == "true"
    )
    theme: str = field(
        default_factory=lambda: os.getenv("UI_THEME", "soft")
    )


# =============================================================================
# Main Settings Container
# =============================================================================

@dataclass
class Settings:
    """Top-level settings container aggregating all configuration sections."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Project metadata
    project_name: str = "SafetyAgent"
    version: str = "0.1.0"
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        
        if not self.llm.api_key:
            warnings.append(
                "LLM_API_KEY not set. Set via environment variable or .env file."
            )
        
        if "your-workspace" in self.llm.api_base_url:
            warnings.append(
                "LLM_API_BASE_URL still has placeholder. Update with your endpoint."
            )
        
        return warnings
    
    def print_status(self) -> None:
        """Print configuration status for debugging."""
        print(f"\n{'='*60}")
        print(f"  {self.project_name} v{self.version} — Configuration")
        print(f"{'='*60}")
        print(f"  Primary Model  : {self.llm.primary_model}")
        print(f"  Fallback Models: {', '.join(self.llm.fallback_models)}")
        print(f"  API Base URL   : {self.llm.api_base_url[:50]}...")
        print(f"  Embedding Model: {self.embedding.model_name}")
        print(f"  Project Root   : {self.paths.root}")
        print(f"  Debug Mode     : {self.debug}")
        print(f"{'='*60}")
        
        warnings = self.validate()
        if warnings:
            print("  ⚠️  Warnings:")
            for w in warnings:
                print(f"     - {w}")
            print()


# =============================================================================
# Singleton Instance
# =============================================================================

settings = Settings()


if __name__ == "__main__":
    settings.print_status()
