DOMAIN = "oswald_conversation"

CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_LLM_HASS_API = "llm_hass_api"
CONF_MAX_HISTORY = "max_history_messages"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_PROMPT = (
    "You are a assistant for Home Assistant.\n"
    "Answer questions about the world truthfully.\n"
    "Answer in plain text. Keep it simple and to the point."
)
DEFAULT_LLM_HASS_API = ["assist"]
DEFAULT_MAX_HISTORY = 20
