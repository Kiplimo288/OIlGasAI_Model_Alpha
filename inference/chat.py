"""
chat.py
-------
Single-turn and multi-turn inference against OilGasAI-Model-Alpha.
Works with both local model and HF Inference API backends.
"""

import os
import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

SYSTEM_PROMPT = """You are OilGasAI Model Alpha, a domain-expert AI assistant built by OilGasAI for the oil and gas industry.
You specialize in EPA GHGRP compliance (Subpart W), methane emissions monitoring, LDAR,
predictive maintenance, SCADA analytics, OGMP 2.0 reporting, and low-cost sensor calibration.
You were trained on 21,429 domain-specific instruction pairs using compute resources from UNDP and CINECA's Leonardo HPC.
Always provide technically accurate, actionable responses. Caveat uncertain information and
recommend professional verification for safety-critical decisions."""


def chat(message: str, backend: dict, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Single-turn chat with OilGasAI Model Alpha.

    Args:
        message: User query string
        backend: Dict from get_inference_backend()
        system_prompt: Optional custom system prompt

    Returns:
        str: Model response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    return _run_inference(messages, backend)


def chat_with_history(
    messages: list[dict],
    backend: dict,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Multi-turn chat preserving conversation history.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
        backend: Dict from get_inference_backend()
        system_prompt: Optional custom system prompt

    Returns:
        str: Model response
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    return _run_inference(full_messages, backend)


def _run_inference(messages: list[dict], backend: dict) -> str:
    """Internal dispatcher between local and API backends."""

    if backend["mode"] == "local":
        return _local_inference(messages, backend["model"], backend["tokenizer"])
    elif backend["mode"] == "api":
        return _api_inference(messages, backend["client"])
    else:
        raise ValueError(f"Unknown backend mode: {backend['mode']}")


def _local_inference(messages: list[dict], model, tokenizer) -> str:
    """Run inference on a locally loaded model."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def _api_inference(messages: list[dict], client) -> str:
    """Run inference via HF Inference API."""
    response = client.chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()
