"""
Configuration file for RAG Application
Centralized settings for easy customization
"""

# --- Model Configuration ---
class ModelConfig:
    """Google Gemini model settings"""
    FLASH_MODEL = "models/gemini-2.0-flash-exp"
    PRO_MODEL = "models/gemini-2.0-flash-thinking-exp-1219"
    DEFAULT_TEMPERATURE = 0.7
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 1.0
    TOP_P = 0.85
    TOP_K = 40
    MAX_OUTPUT_TOKENS = 2048

# --- Document Processing Configuration ---
class DocumentConfig:
    """PDF processing and chunking settings"""
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_SEARCH_K = 4  # Number of chunks to retrieve

# --- Embedding Configuration ---
class EmbeddingConfig:
    """Embedding model settings"""
    MODEL_NAME = "all-MiniLM-L6-v2"
    CACHE_FOLDER = "./models"

# --- UI Configuration ---
class UIConfig:
    """User interface settings"""
    PAGE_TITLE = "Chat with PDF - Scholar AI"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Gradient colors
    GRADIENT_START = "#667eea"
    GRADIENT_END = "#764ba2"
    
    # Chat history
    MAX_CHAT_HISTORY_DISPLAY = 10  # Number of messages to include in context

# --- Feature Flags ---
class FeatureFlags:
    """Enable/disable features"""
    ENABLE_FEEDBACK = True
    ENABLE_TOKEN_TRACKING = True
    ENABLE_TEMPERATURE_CONTROL = True
    ENABLE_CONVERSATION_SUMMARY = True
    ENABLE_DOCUMENT_FILTER = True
    ENABLE_STREAMING = True
    ENABLE_EXPORT = True

# --- Logging Configuration ---
class LogConfig:
    """Logging settings"""
    LOG_FILE = "app.log"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- API Configuration ---
class APIConfig:
    """API usage and limits"""
    # Approximate token costs (adjust based on actual pricing)
    COST_PER_1K_INPUT_TOKENS = 0.00015  # USD
    COST_PER_1K_OUTPUT_TOKENS = 0.0006  # USD
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    
    # Token estimation (rough approximation)
    CHARS_PER_TOKEN = 4

# --- Session Configuration ---
class SessionConfig:
    """Session management settings"""
    SESSION_SAVE_DIR = "./sessions"
    AUTO_SAVE = False
    MAX_SESSIONS = 10
