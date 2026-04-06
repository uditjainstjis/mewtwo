import abc
from typing import List, Dict, Any, Optional

class BaseEngine(abc.ABC):
    """
    Abstract base class for all inference engines (e.g., HF CUDA, vLLM).
    Handles prompt generation, composition, and perplexity calculation.
    """

    @abc.abstractmethod
    def __init__(self, model_id: str, **kwargs):
        """Initialize the model and tokenizer."""
        pass

    @abc.abstractmethod
    def generate(self,
                 prompts: List[str],
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 active_adapters: Optional[Dict[str, float]] = None,
                 composition_method: str = "additive",
                 **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input strings.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            active_adapters: Dict mapping adapter_name -> float routing weight.
            composition_method: Strategy used to compose multiple adapters.
            
        Returns:
            List of generated strings.
        """
        pass

    @abc.abstractmethod
    def compute_perplexity(self,
                           text: str,
                           active_adapters: Optional[Dict[str, float]] = None,
                           composition_method: str = "additive") -> float:
        """
        Calculate perplexity of the given text sequence.
        """
        pass

    @abc.abstractmethod
    def load_adapter(self, adapter_name: str, adapter_path: str) -> None:
        """
        Load a LoRA adapter from disk into the engine's memory.
        """
        pass
        
    @abc.abstractmethod
    def unload_adapter(self, adapter_name: str) -> None:
        """
        Unload an adapter to free up memory.
        """
        pass
