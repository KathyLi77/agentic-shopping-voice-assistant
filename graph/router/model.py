# graph/router/model.py
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import logging
logger = logging.getLogger(__name__)

def load_router_model():
    """Load Qwen3-4B-Instruct (July 2025 release)."""
    
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise
    
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        raise
    
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("Model loaded successfully")
    
    return llm
