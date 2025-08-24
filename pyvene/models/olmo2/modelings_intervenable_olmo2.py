"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..constants import *  # expects CONST_INPUT_HOOK / CONST_OUTPUT_HOOK, split_head_and_permute, etc.


"""olmo2 module path mappings"""
olmo2_type_to_module_mapping = {
    # Block-level hooks
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),

    # MLP
    "mlp_activation": ("layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),

    # Attention outputs
    # NOTE: `attention_value_output` hooks the input of o_proj (the combined value stream after attn weights).
    "attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "layers[%s].self_attn.o_proj",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "n_head"),
    ),

    # Attention module I/O
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),

    # Projections (pre-head-reshape tensors)
    "query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),

    # Per-head versions (split and permute using provided helper)
    "head_query_output": (
        "layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_head"),
    ),
    "head_key_output": (
        "layers[%s].self_attn.k_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_kv_head"),
    ),
    "head_value_output": (
        "layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_kv_head"),
    ),
}


"""olmo2 dimension mappings"""
olmo2_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "n_kv_head": ("num_key_value_heads",),

    # Block-level sizes
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),

    # MLP sizes
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),

    # Attention shapes
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),

    # Projections
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),

    # Per-head projections (default to hidden_size/num_attention_heads for head_dim)
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
}


"""olmo2 model with LM head"""
olmo2_lm_type_to_module_mapping = {}
for k, v in olmo2_type_to_module_mapping.items():
    olmo2_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

olmo2_lm_type_to_dimension_mapping = olmo2_type_to_dimension_mapping

"""olmo2 model with classifier head"""
olmo2_classifier_type_to_module_mapping = {}
for k, v in olmo2_type_to_module_mapping.items():
    olmo2_classifier_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

olmo2_classifier_type_to_dimension_mapping = olmo2_type_to_dimension_mapping


def create_olmo2(
    name="meta-olmo2/Olmo2-2-7b-hf",
    cache_dir=None,
    dtype=torch.bfloat16,
    config=None,
    revision="main",
):
    """
    Creates an Olmo-2 Causal LM model, config, and tokenizer from the given name and revision.

    Returns:
        (config, tokenizer, model)
    """
    if config is None:
        config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            revision=revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, revision=revision)
    else:
        model = AutoModelForCausalLM.from_config(config, cache_dir=cache_dir, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, revision=revision)

    print("loaded Olmo-2 model")
    return config, tokenizer, model
