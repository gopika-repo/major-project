import os
from dotenv import load_dotenv

# This loads your .env file so Python can read your API keys
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class Settings:
    def __init__(self):
        # API Keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

        # Gemini Model Names
        self.MODEL_NAME = "gemini-3.5-flash"
        self.EMBED_MODEL = "models/gemini-embedding-001"
        self.EMBED_DIMENSIONS = 3072

        # Qdrant Settings
        self.QDRANT_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qdrant_storage"
        )
        self.QDRANT_COLLECTION = "financial_rag"

        # RAG Settings
        self.CHUNK_SIZE = 400
        self.CHUNK_OVERLAP = 50
        self.TOP_K = 5

        # File Upload Settings
        self.MAX_FILE_SIZE_MB = 50
        self.UPLOAD_DIR = "backend/data/uploads"
        self.EXTRACTED_DIR = "backend/data/extracted"
        self.OUTPUT_DIR = "backend/data/outputs"

    def validate(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing in .env file")
        print("✅ Configuration loaded successfully")

    def display(self):
        print(f"MODEL      : {self.MODEL_NAME}")
        print(f"EMBED      : {self.EMBED_MODEL}")
        print(f"DIMENSIONS : {self.EMBED_DIMENSIONS}")
        print(f"COLLECTION : {self.QDRANT_COLLECTION}")
        print(f"CHUNK_SIZE : {self.CHUNK_SIZE}")
        print(f"TOP_K      : {self.TOP_K}")


# Single instance used across entire project
settings = Settings()


if __name__ == "__main__":
    settings.validate()
    settings.display()