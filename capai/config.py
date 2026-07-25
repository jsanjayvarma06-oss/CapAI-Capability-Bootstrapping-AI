import os
from pathlib import Path

# ---------------------------------------------------------------- Paths
CAPAI_HOME = Path(os.environ.get("CAPAI_HOME", Path.home() / ".capai"))
CAPAI_HOME.mkdir(parents=True, exist_ok=True)

MCP_SERVERS_DIR = Path(os.environ.get("MCP_SERVERS_DIR", CAPAI_HOME / "mcp_servers"))
MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- MongoDB
MONGODB_URI = os.environ.get("MONGODB_URI")

# ---------------------------------------------------------------- Sandbox
SANDBOX_TIMEOUT_SECONDS  = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", 10))
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("MAX_ACQUISITION_ATTEMPTS", 3))
MIN_TEST_CASES           = int(os.environ.get("MIN_TEST_CASES", 3))
COVERAGE_THRESHOLD       = float(os.environ.get("COVERAGE_THRESHOLD", 70.0))
MAX_CODE_LENGTH          = int(os.environ.get("MAX_CODE_LENGTH", 50000))
MAX_DESCRIPTION_LENGTH   = int(os.environ.get("MAX_DESCRIPTION_LENGTH", 2000))

# ---------------------------------------------------------------- Registry
SIMILARITY_THRESHOLD     = float(os.environ.get("SIMILARITY_THRESHOLD", 0.7))
MAX_REGISTRY_SIZE        = int(os.environ.get("MAX_REGISTRY_SIZE", 10000))
CAPABILITY_TTL_DAYS      = int(os.environ.get("CAPABILITY_TTL_DAYS", 0))  # 0 = never expire

# ---------------------------------------------------------------- MCP
CAPAI_MCP_ALLOWED_HOSTS  = os.environ.get(
    "CAPAI_MCP_ALLOWED_HOSTS",
    "capai-capability-bootstrapping-ai-fu58.onrender.com"
)

# ---------------------------------------------------------------- LLM
NVIDIA_API_KEY           = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL             = os.environ.get("CAPAI_NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_ENABLE_THINKING   = os.environ.get("CAPAI_NVIDIA_THINKING", "false").lower() == "true"
NVIDIA_REASONING_BUDGET  = int(os.environ.get("CAPAI_NVIDIA_REASONING_BUDGET", "4096"))

GROQ_API_KEY             = os.environ.get("GROQ_API_KEY")
GROQ_MODEL               = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")

CEREBRAS_API_KEY         = os.environ.get("CEREBRAS_API_KEY")
CEREBRAS_MODEL           = os.environ.get("CAPAI_CEREBRAS_MODEL", "gpt-oss-120b")

ANTHROPIC_API_KEY        = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL          = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

LLM_ENABLED              = bool(NVIDIA_API_KEY or GROQ_API_KEY)

# ---------------------------------------------------------------- Skill endpoint
SKILL_MAX_FILES          = int(os.environ.get("SKILL_MAX_FILES", 15))
SKILL_MAX_FILE_SIZE_KB   = int(os.environ.get("SKILL_MAX_FILE_SIZE_KB", 50))
GITHUB_TOKEN             = os.environ.get("GITHUB_TOKEN")
