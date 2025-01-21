from typing import Any

class ModelInterface:
    """Abstract base class for model inference."""
    def infer(self, prompt: str) -> Any:
        raise NotImplementedError("The infer method must be implemented by subclasses.")