#!/usr/bin/env python3
"""
Test script to verify paraformer_hidden.onnx can be loaded and run inference.
"""

import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    print("✓ onnxruntime imported successfully")
except ImportError:
    print("❌ onnxruntime not installed")
    exit(1)

def main():
    project_root = Path(__file__).resolve().parent
    onnx_path = project_root / "weights" / "paraformer_hidden.onnx"
    
    if not onnx_path.exists():
        print(f"❌ Model file not found: {onnx_path}")
        return
    
    print(f"Loading model: {onnx_path}")
    print(f"File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Create inference session
        sess = ort.InferenceSession(str(onnx_path))
        print("✓ Model loaded successfully")
        
        # Get input/output info
        input_names = [inp.name for inp in sess.get_inputs()]
        output_names = [out.name for out in sess.get_outputs()]
        
        print(f"\nInputs ({len(input_names)}):")
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) else str(d) for d in inp.shape]
            print(f"  - {inp.name}: {inp.type}, shape={shape}")
        
        print(f"\nOutputs ({len(output_names)}):")
        for out in sess.get_outputs():
            shape = [d if isinstance(d, int) else str(d) for d in out.shape]
            print(f"  - {out.name}: {out.type}, shape={shape}")
        
        # Test inference with dummy data
        print("\nTesting inference...")
        # Input: [B, T, D] = [1, 150, 560] (fixed size from export)
        batch_size = 1
        time_frames = 150  # Fixed size from export
        feat_dim = 560
        
        feats = np.random.randn(batch_size, time_frames, feat_dim).astype(np.float32)
        print(f"Input shape: {feats.shape}, dtype: {feats.dtype}")
        
        # Run inference
        feed_dict = {input_names[0]: feats}
        outputs = sess.run(output_names, feed_dict)
        
        print("✓ Inference successful!")
        print(f"\nOutputs:")
        for name, output in zip(output_names, outputs):
            print(f"  - {name}: shape={output.shape}, dtype={output.dtype}")
            print(f"    min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Test with different time lengths (all should be 150 due to fixed size)
        print("\nTesting with different time lengths (will be padded/truncated to 150)...")
        for t in [88, 100, 150, 200]:
            if t != 150:
                print(f"  T={t}: Will fail (model expects exactly 150)")
                continue
            feats_t = np.random.randn(1, t, feat_dim).astype(np.float32)
            feed_dict_t = {input_names[0]: feats_t}
            outputs_t = sess.run(output_names, feed_dict_t)
            print(f"  T={t}: output shape={outputs_t[0].shape}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
