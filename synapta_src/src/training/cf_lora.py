import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Trainer

class CFLoRATrainer(Trainer):
    """
    Composition-Friendly LoRA (CF-LoRA) Trainer.
    
    Novel contribution: Penalizes the adapter being trained for overlapping
    with the subspaces of PREVIOUSLY trained adapters.
    
    Loss = Loss_Task + λ * Σ_j || P_j @ ΔW_current ||²_F / || ΔW_current ||²_F
    """
    def __init__(self, *args, frozen_subspaces=None, lambda_ortho=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_subspaces = frozen_subspaces or []
        self.lambda_ortho = lambda_ortho
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Standard task loss (Language Modeling)
        outputs = model(**inputs)
        lm_loss = outputs.loss
        
        # 2. If no prior adapters, standard LoRA
        if not self.frozen_subspaces or self.lambda_ortho == 0:
            return (lm_loss, outputs) if return_outputs else lm_loss
            
        # 3. Compute Orthogonality Penalty
        ortho_loss = 0.0
        n_layers = 0
        
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                layer_base = name.replace('.lora_A.default.weight', '')
                A = param # [r, d_in]
                B_name = name.replace('lora_A', 'lora_B')
                
                # Retrieve B matrix securely
                param_dict = dict(model.named_parameters())
                if B_name in param_dict:
                    B = param_dict[B_name] # [d_out, r]
                    
                    # Compute current update: ΔW = B @ A
                    delta_w = B @ A
                    dw_norm_sq = torch.norm(delta_w, p='fro')**2 + 1e-8
                    
                    # Penalize overlap with every prior adapter's subspace
                    layer_overlap = 0.0
                    for subspace_dict in self.frozen_subspaces:
                        if layer_base in subspace_dict:
                            P_j_cpu = subspace_dict[layer_base] # Retrieve from RAM
                            # Push to GPU for this specific math operation only
                            P_j_gpu = P_j_cpu.to(delta_w.device, non_blocking=True)
                            
                            # Projected component
                            projected = P_j_gpu @ delta_w
                            overlap = torch.norm(projected, p='fro')**2
                            layer_overlap += overlap / dw_norm_sq
                            
                            # Release from VRAM immediately!
                            del P_j_gpu
                    ortho_loss += layer_overlap
                    n_layers += 1
                    
        if n_layers > 0:
            ortho_loss = ortho_loss / n_layers
            
        # Final combined loss
        total_loss = lm_loss + self.lambda_ortho * ortho_loss
        
        # Log to wandb/tensorboard if enabled
        if self.state.global_step % 10 == 0:
            self.log({"cf_lora/ortho_loss": ortho_loss.item(), "cf_lora/lm_loss": lm_loss.item()})
            
        return (total_loss, outputs) if return_outputs else total_loss
