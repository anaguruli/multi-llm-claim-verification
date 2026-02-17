import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration
DECOMPOSER_MODEL = "gemini-2.5-flash"
VERIFIER_MODELS = {
    "v1": "gemini-2.5-flash",
    "v2": "gemini-2.5-flash",
    "v3": "gemini-2.5-flash",
}
SYNTHESIZER_MODEL = "gemini-2.5-flash"
BASELINE_MODEL = "gemini-2.5-flash"

# RAG configuration
CHUNK_SIZE = 400          # words per chunk
CHUNK_OVERLAP = 80        # words overlap between chunks
TOP_K_RETRIEVAL = 4       # passages retrieved per sub-claim
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Verifier temperatures (different per verifier for diversity)
VERIFIER_TEMPERATURES = {
    "v1": 0.1,   # Conservative / strict
    "v2": 0.7,   # Moderate
    "v3": 0.4,   # Balanced
}

DECOMPOSER_TEMPERATURE = 0.2
SYNTHESIZER_TEMPERATURE = 0.2
BASELINE_TEMPERATURE = 0.3

# Paths
DATA_DIR = "data"
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [DATA_DIR, KB_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)