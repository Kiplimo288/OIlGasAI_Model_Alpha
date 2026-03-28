"""
load_model.py
-------------
Loads OilGasAI-Model-Alpha either locally (4-bit QLoRA) or via the
Hugging Face Inference API, depending on the USE_HF_API env flag.
"""

import os
import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "OilgasAI/OilGasAI-Model-Alpha")
QUANTIZATION = os.getenv("QUANTIZATION", "4bit")
USE_HF_API = os.getenv("USE_HF_API", "false").lower() == "true"


def load_model():
    """
    Load OilGasAI Model Alpha locally using 4-bit or 8-bit quantization via bitsandbytes.
    Requires a GPU with sufficient VRAM (min ~40GB for 4-bit 70B).

    Returns:
        tuple: (model, tokenizer)
    """
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set in .env file.")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    logger.info(f"Loading OilGasAI-Model-Alpha locally ({QUANTIZATION} quantization)...")

    # --- Quantization config ---
    if QUANTIZATION == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif QUANTIZATION == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    # --- Load base model ---
    base_model_id = "meta-llama/Llama-3.1-70B-Instruct"
    logger.info(f"Loading base model: {base_model_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )

    # --- Load OilGasAI Model Alpha LoRA adapters ---
    logger.info(f"Loading LoRA adapters from: {HF_MODEL_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        HF_MODEL_ID,
        token=HF_TOKEN,
    )
    model.eval()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("OilGasAI-Model-Alpha loaded successfully.")
    return model, tokenizer


def load_hf_client():
    """
    Returns an InferenceClient for the HF Serverless Inference API.
    Use this when you don't have local GPU resources.

    Returns:
        huggingface_hub.InferenceClient
    """
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set in .env file.")

    from huggingface_hub import InferenceClient

    logger.info(f"Connecting to HF Inference API: {HF_MODEL_ID}")
    client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)
    return client


def get_inference_backend():
    """
    Returns the appropriate backend based on USE_HF_API flag.
    Used by the API layer to stay backend-agnostic.

    Returns:
        dict with keys: 'mode', 'model' (and 'tokenizer' if local)
    """
    if USE_HF_API:
        client = load_hf_client()
        return {"mode": "api", "client": client}
    else:
        model, tokenizer = load_model()
        return {"mode": "local", "model": model, "tokenizer": tokenizer}
