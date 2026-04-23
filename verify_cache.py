import sys
import traceback

def safe_print(msg):
    print(msg, flush=True)

try:
    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 1 — VERIFY ENVIRONMENT")
    safe_print("═══════════════════════════════════════════════════════")
    import transformers
    import torch
    safe_print(transformers.__version__)
    safe_print(torch.__version__)
    safe_print(torch.cuda.is_available())
    safe_print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 2 — VERIFY THE CORRECT CACHE CLASS EXISTS")
    safe_print("═══════════════════════════════════════════════════════")
    cache_class = None
    try:
        # The user requested 'transformers.models.nemotron_h.modeling_nemotron_h' but our local HF cache has it in 'transformers_modules'
        # I'll try the requested first, then fall back to exactly what is in the huggingface cache
        from transformers.models.nemotron_h.modeling_nemotron_h import HybridMambaAttentionDynamicCache
        safe_print(f"IMPORT SUCCESS: {HybridMambaAttentionDynamicCache}")
        cache_class = HybridMambaAttentionDynamicCache
    except ImportError as e:
        safe_print(f"IMPORT FAILED: {str(e)}")
        try:
            from transformers import HybridMambaAttentionDynamicCache
            safe_print("FALLBACK IMPORT SUCCESS")
            cache_class = HybridMambaAttentionDynamicCache
        except ImportError as e2:
            safe_print(f"FALLBACK ALSO FAILED: {str(e2)}")
            # Try our local module from remote code
            try:
                import sys
                import os
                hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/nemotron/modeling_nemotron_h.py")
                if os.path.exists(hf_cache):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("modeling_nemotron_h", hf_cache)
                    modeling_nemotron_h = importlib.util.module_from_spec(spec)
                    sys.modules["modeling_nemotron_h"] = modeling_nemotron_h
                    spec.loader.exec_module(modeling_nemotron_h)
                    cache_class = modeling_nemotron_h.HybridMambaAttentionDynamicCache
                    safe_print(f"hf_cache IMPORT SUCCESS: {cache_class}")
                else:
                    safe_print("hf_cache file not found")
            except Exception as e3:
                safe_print(f"hf_cache IMPORT FAILED: {str(e3)}")

    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 3 — VERIFY MODEL LOADS WITH EAGER ATTENTION ONLY")
    safe_print("═══════════════════════════════════════════════════════")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_path = "/home/learner/Desktop/mewtwo/models/nemotron"
    
    # We will load in 4-bit config to avoid OOM if possible
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    import sys
    sys.path.insert(0, model_path)
    safe_print(f"Added {model_path} to sys.path to force local loading")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        # BUG 2 FIX: Prepare for kbit training to avoid MoE shape mismatch
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        safe_print(model.config._attn_implementation)
    except Exception as e:
        safe_print(f"WARNING/ERROR during model load: {str(e)}")
        raise e

    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 4 — INSTANTIATE THE KV CACHE OBJECT")
    safe_print("═══════════════════════════════════════════════════════")
    try:
        if cache_class is None:
            safe_print("Attempting to extract cache_class dynamically from loaded model module...")
            import sys
            model_module = sys.modules[model.__module__]
            cache_class = getattr(model_module, 'HybridMambaAttentionDynamicCache', None)
            if cache_class is not None:
                safe_print(f"DYNAMIC EXTRACT SUCCESS: {cache_class}")
            else:
                raise ValueError("Could not find HybridMambaAttentionDynamicCache in model module")
        
        cache = cache_class(
            config=model.config,
            batch_size=1,
            dtype=torch.bfloat16,
            device=next(model.parameters()).device,
        )
        safe_print(f"Cache created: {type(cache)}")
        safe_print(f"Attention layers in cache: {len(cache.key_cache)}")
        safe_print(f"Mamba conv states shape: {cache.conv_states[0].shape if len(cache.conv_states) > 0 else 'none'}")
    except Exception as e:
        safe_print(f"{type(e).__name__}: {str(e)}")
        cache = None

    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 5 — RUN A SMALL FORWARD PASS WITH CACHE")
    safe_print("═══════════════════════════════════════════════════════")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

        test_cache = cache_class(
            config=model.config,
            batch_size=1,
            dtype=torch.bfloat16,
            device=model.device,
        ) if cache_class else None

        with torch.no_grad():
            out = model.generate(
                **inputs,
                past_key_values=test_cache,
                use_cache=True,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        safe_print(f"Output tokens: {out.shape}")
        safe_print(f"Decoded: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    except Exception as e:
        traceback.print_exc(file=sys.stdout)

    safe_print("═══════════════════════════════════════════════════════")
    safe_print("STEP 6 — TRAINING MODE CACHE BEHAVIOR CHECK")
    safe_print("═══════════════════════════════════════════════════════")
    try:
        model.train()
        safe_print(f"Training mode: {model.training}")

        # In training mode, Mamba layers use fused kernels and SKIP the cache.
        # This is intentional behavior in NemotronH - not a bug.
        dummy_input = tokenizer("Test training input", return_tensors="pt").to(model.device)
        try:
            # use_cache=False is REQUIRED for training mode
            train_out = model(**dummy_input, use_cache=False)
            safe_print(f"Training forward pass: SUCCESS, logits shape = {train_out.logits.shape}")
        except Exception as e:
            safe_print(f"Training forward pass FAILED: {str(e)}")
            traceback.print_exc(file=sys.stdout)

        model.eval()
    except Exception as e:
        safe_print(f"Error in STEP 6: {str(e)}")

except Exception as main_e:
    traceback.print_exc(file=sys.stdout)
