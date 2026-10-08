import torch 
import math 
from transformers import PreTrainedModel

def load_model(model_id, seqlen) -> PreTrainedModel:
    """Load only LLM"""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    ctx = getattr(config, "max_position_embeddings", None)
    if ctx and seqlen > ctx:
        config.rope_scaling = {"type": "linear", "factor": float(math.ceil(seqlen / ctx))}
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_safetensors=True,
                                                dtype=torch.bfloat16, device_map="auto")
    return model.eval().bfloat16()
