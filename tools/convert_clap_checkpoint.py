#!/usr/bin/env python3
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for direct checkpoint conversion and validation by the Gangqinpu authors.
# Licensed under the Apache License, Version 2.0; see LICENSES/Apache-2.0.txt.
# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Convert the official LAION-CLAP music checkpoint to a Hugging Face ``ClapModel``.

The ``laion/larger_clap_music`` weights downloaded during construction produced
collapsed text embeddings (see https://huggingface.co/laion/larger_clap_music/discussions/2),
so step 7 uses the upstream checkpoint ``music_audioset_epoch_15_esc_90.14.pt``
(https://huggingface.co/lukewys/laion_clap) converted with the key mapping of
the Transformers v4.35.0 CLAP converter. Architecture config and processor are
taken from ``laion/larger_clap_music``.

    python tools/convert_clap_checkpoint.py --checkpoint music_audioset_epoch_15_esc_90.14.pt \
        --output models/clap_music_converted

The renaming function below is from the HuggingFace Transformers CLAP
converter, Copyright 2023 The HuggingFace Inc. team, Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import ClapConfig, ClapModel, ClapProcessor

KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}


def rename_state_dict(state_dict: dict) -> dict:
    model_state_dict = {}
    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if re.match(sequential_layers_pattern, key):
            layer = re.match(sequential_layers_pattern, key).group(1)
            key = key.replace(f"sequential.{layer}.", f"layers.{int(layer) // 3}.linear.")
        elif re.match(text_projection_pattern, key):
            layer = int(re.match(text_projection_pattern, key).group(1))
            key = key.replace(f"_projection.{layer}.", f"_projection.linear{1 if layer == 0 else 2}.")
        if "qkv" in key:  # equivalent to the upstream converter condition
            qkv_dim = value.size(0) // 3
            model_state_dict[key.replace("qkv", "query")] = value[:qkv_dim]
            model_state_dict[key.replace("qkv", "key")] = value[qkv_dim:qkv_dim * 2]
            model_state_dict[key.replace("qkv", "value")] = value[qkv_dim * 2:]
        else:
            model_state_dict[key] = value
    return model_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--config", default="laion/larger_clap_music", help="HF repo or folder for config/processor")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = rename_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()})
    model = ClapModel(ClapConfig.from_pretrained(args.config)).eval()
    report = model.load_state_dict(state, strict=False)
    if not set(report.missing_keys) <= {"text_model.embeddings.token_type_ids"}:
        raise RuntimeError(f"incomplete conversion, missing: {report.missing_keys}")
    unused = ["spectrogram_extractor", "logmel_extractor", "attn_mask", "position_ids", "text_transform.",
              "audio_transform.", "tscam_conv.", "audio_encoder.head."]
    if not all(any(t in k for t in unused) for k in report.unexpected_keys):
        raise RuntimeError(f"unexpected keys: {report.unexpected_keys}")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    processor = ClapProcessor.from_pretrained(args.config)
    processor.save_pretrained(args.output)

    with torch.inference_mode():
        tokens = processor(text=["A dog barking.", "Heavy metal music.", "Classical piano music."],
                           padding=True, return_tensors="pt")
        text = torch.nn.functional.normalize(model.get_text_features(**tokens), dim=-1)
        cosines = (text @ text.T).tolist()
    if cosines[0][1] > 0.95 or cosines[0][2] > 0.95:
        raise RuntimeError("text embeddings still collapsed; conversion failed")
    (args.output / "conversion_report.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint.name, "missing_keys": report.missing_keys,
         "unexpected_keys": report.unexpected_keys, "text_cosines": cosines}, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
