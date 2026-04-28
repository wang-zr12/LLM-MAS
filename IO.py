import asyncio
import  json
import os

class ResultWriter:
    """Thread-safe file writer"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.lock = asyncio.Lock()
        self.file = None

    async def __aenter__(self):
        self.file = open(self.file_path, 'a', encoding='utf-8')
        return self

    async def write_result(self, result):
        async with self.lock:
            self.file.write(json.dumps(result, ensure_ascii=False) + '\n')
            self.file.flush()
            os.fsync(self.file.fileno())

    async def __aexit__(self, exc_type, exc, tb):
        if self.file:
            self.file.close()

def load_file(file_path: str):
    """Load JSON file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load character profiles: {str(e)}")
