import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .base_engine import BaseEngine

class HFEngine(BaseEngine):
    def __init__(self, model_id: str,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda",
                 gradient_checkpointing: bool = False,
                 **kwargs):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        print(f"Loading HFEngine base model: {model_id} onto {device} in {dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            **kwargs
        )
        # Gradient checkpointing helps save VRAM if we want to run higher ranks or larger batches
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Wrap with PEFT right away to be ready
        # peft >= 0.14 allows adding multiple adapters and activating/deactivating them dynamically
        from peft import inject_adapter_in_model, LoraConfig
        self.model = PeftModel(self.model, LoraConfig(task_type="CAUSAL_LM"), adapter_name="default")
        self.adapters_loaded = {"default"}

    def load_adapter(self, adapter_name: str, adapter_path: str) -> None:
        if adapter_name in self.adapters_loaded:
            return
        print(f"Loading adapter '{adapter_name}' from {adapter_path}...")
        self.model.load_adapter(adapter_path, adapter_name=adapter_name)
        self.adapters_loaded.add(adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        if adapter_name in self.adapters_loaded and adapter_name != "default":
            self.model.delete_adapter(adapter_name)
            self.adapters_loaded.remove(adapter_name)

    def generate(self,
                 prompts: List[str],
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 active_adapters: Optional[Dict[str, float]] = None,
                 composition_method: str = "additive",
                 **kwargs) -> List[str]:
        """
        Generate using HuggingFace model.
        In HF, prompt-level merging involves setting active adapters and weights prior to forward pass.
        Future modules will implement advanced SAC logic modifying this base pipeline.
        """
        # Set active adapters if specified
        if active_adapters:
            adapters = list(active_adapters.keys())
            weights = list(active_adapters.values())
            self.model.set_adapter(adapters)
            # PEFT allows weighted adapter setting
            # We'll use this as the baseline additive composition
        else:
            self.model.disable_adapters()

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        
        responses = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if active_adapters:
            # Cleanup state
            self.model.disable_adapters()
            
        return responses

    def compute_perplexity(self, text: str,
                           active_adapters: Optional[Dict[str, float]] = None,
                           composition_method: str = "additive") -> float:
        if active_adapters:
            self.model.set_adapter(list(active_adapters.keys()))
        else:
            self.model.disable_adapters()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            ppl = torch.exp(loss)
            
        if active_adapters:
            self.model.disable_adapters()
            
        return ppl.item()
