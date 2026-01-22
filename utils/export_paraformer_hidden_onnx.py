#!/usr/bin/env python3
"""
Export Paraformer encoder hidden states to ONNX using ParaformerHiddenWrapper.

Output:
    weights/paraformer_hidden.onnx (+ external data file if large)

Usage:
    python export_paraformer_hidden_onnx.py
"""

import os
from pathlib import Path

import torch

from funasr_local.tasks.asr import ASRTaskParaformer as ASRTask
from extract_paraformer_feature import ParaformerHiddenWrapperSingleInput


def main():
    project_root = Path(__file__).resolve().parent

    # Paths to Paraformer weights (same as used by backend feature extraction)
    model_dir = project_root / "weights" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    asr_train_config = model_dir / "config.yaml"
    asr_model_file = model_dir / "model.pb"
    cmvn_file = model_dir / "am.mvn"

    assert asr_train_config.is_file(), f"config not found: {asr_train_config}"
    assert asr_model_file.is_file(), f"model.pb not found: {asr_model_file}"
    assert cmvn_file.is_file(), f"cmvn file not found: {cmvn_file}"

    device = "cpu"

    print("Loading Paraformer model...")
    asr_model, _ = ASRTask.build_model_from_file(
        str(asr_train_config),
        str(asr_model_file),
        str(cmvn_file),
        device=device,
    )
    asr_model.to(device=device, dtype=torch.float32).eval()

    # Use single-input wrapper (automatically computes feats_lengths from feats shape)
    wrapper = ParaformerHiddenWrapperSingleInput(asr_model).to(device).eval()

    # Dummy input: frontend features (after LFR). With n_mels=80 and lfr_m=7 -> D=560.
    # This bypasses wav_frontend FFT so ONNX export won't hit aten::fft_rfft.
    # NOTE: Paraformer encoder has hardcoded attention mask dimensions based on dummy input size.
    # We use a fixed size (150) that should cover most use cases. If input is longer, it will be truncated.
    feat_dim = 80 * 7
    dummy_t = 150  # Fixed size for attention masks. Inputs longer than this will fail.
    feats = torch.randn(1, dummy_t, feat_dim, dtype=torch.float32, device=device)
    # Note: feats_lengths is now computed automatically inside the wrapper

    out_dir = project_root / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "paraformer_hidden.onnx"

    print(f"Exporting ONNX to: {onnx_path}")
    print("Using single-input wrapper (feats_lengths computed automatically)")
    print(f"Dummy input size: [1, {dummy_t}, {feat_dim}]")

    torch.onnx.export(
        wrapper,
        feats,  # Only pass feats, feats_lengths is computed inside
        str(onnx_path),
        input_names=["feats"],
        output_names=["hidden"],
        dynamic_axes={
            "feats": {0: "batch", 1: "time"},
            "hidden": {0: "batch", 2: "time"},
        },
        opset_version=17,
        do_constant_folding=False,  # Disable constant folding to preserve dynamic dimensions
        # NOTE: current torch version in your env does not support `use_external_data_format`.
        # If the exported model is >2GB, you can later split it with ONNX tools.
    )

    print("Done.")
    print(f"ONNX model saved to: {onnx_path}")
    print("Note: if the model is larger than 2GB, weights will be stored in a separate .onnx_data file next to the ONNX file.")
    print(f"\n⚠️  IMPORTANT: This model is exported with fixed time dimension {dummy_t}.")
    print(f"   Input sequences longer than {dummy_t} frames will cause runtime errors.")
    print(f"   Frontend should truncate or pad inputs to exactly {dummy_t} frames.")


if __name__ == "__main__":
    main()

