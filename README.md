# LiteAvatar - Browser Version

A lightweight audio-driven 2D avatar solution that runs **entirely in the browser** using WebGPU/WASM. No backend server required.

> **Note**: This project is a browser-optimized fork of [HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar). The original project requires Python backend and GPU acceleration. This version solves the problem of running LiteAvatar entirely in the browser without any backend dependencies.

## Highlights & Architecture

- **100% Frontend**: All processing runs in the browser using ONNX Runtime Web (WebGPU/WASM)
- **Full Frontend Feature Extraction**: Complete Paraformer feature extraction pipeline in browser (fbank + LFR + CMVN), using `weights/paraformer_hidden.onnx` (603MB FP32)
- **Static Hosting**: Can be deployed to GitHub Pages, CDN, or any static hosting service
- **No Backend Required**: Completely self-contained, no server-side dependencies

## Models

> Models are large (>100MB) and excluded from Git. You need to download them manually.

### Required Models

- `weights/paraformer_hidden.onnx` (603MB FP32) - Paraformer encoder for feature extraction
- `weights/model_1.onnx` - Audio2mouth model
- `data/preload/net_encode.onnx` - Encoder model
- `data/preload/net_decode.onnx` - Decoder model

### Model Download

Models can be downloaded from:

- **Hugging Face** (Recommended): [fushengji/lite-avatar-wasm](https://huggingface.co/fushengji/lite-avatar-wasm)
  - `paraformer_hidden.onnx` (633MB)
  - `model_1.onnx` (184MB)

You can also download models from ModelScope or use the export scripts (see Development section below).

## Quick Start

### Local Preview

```bash
# Start a simple HTTP server
python -m http.server 8000

# Open http://localhost:8000 in your browser
```

### Deploy to GitHub Pages

1. Push your code to a GitHub repository
2. Go to Settings → Pages
3. Select your branch and `/ (root)` folder
4. Your site will be available at `https://yourusername.github.io/repository-name`

### Deploy to Other Static Hosting

- **Vercel**: Connect your repository, it will auto-detect static files
- **Netlify**: Drag and drop your folder or connect via Git
- **Cloudflare Pages**: Connect repository and set build command to `echo "No build needed"`

## Frontend Usage

1. Ensure all model files are available in the correct paths (or update paths in `js/config.js`)
2. Open the page in your browser
3. Upload audio file (or use default sample audio / microphone recording)
4. Upload avatar data (or use default sample data)
5. Click "Generate Video" and wait for rendering

## Data

- Sample data: `data/preload/` contains sample avatar data (`bg_video.mp4`, `neutral_pose.npy`, `face_box.txt`, `ref_frames/`, etc.)
- More avatars: [LiteAvatarGallery](https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/summary)

## Project Layout

```
index.html                      # Main HTML file
js/
  ├── lite-avatar-web.js        # Main frontend logic
  ├── paraformer-frontend.js    # Paraformer feature extraction (fbank + LFR + CMVN)
  └── config.js                 # Configuration
weights/
  ├── paraformer_hidden.onnx   # Paraformer encoder (603MB)
  └── model_1.onnx             # Audio2mouth model
data/preload/
  ├── net_encode.onnx          # Encoder model
  ├── net_decode.onnx          # Decoder model
  └── ...                      # Sample avatar data
```

## Development

### Export Paraformer Model to ONNX

The `utils/` directory contains scripts for exporting the Paraformer model from PyTorch to ONNX format.

#### Prerequisites

```bash
# Install dependencies
pip install torch onnxruntime funasr
```

#### Export Process

1. **Prepare Model Files**

   Ensure you have the Paraformer model files in the correct location:
   ```
   weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
   ├── config.yaml
   ├── model.pb
   └── am.mvn
   ```

2. **Export to ONNX**

   Run the export script:
   ```bash
   cd utils
   python export_paraformer_hidden_onnx.py
   ```

   This script will:
   - Load the Paraformer model from PyTorch format
   - Wrap it with `ParaformerHiddenWrapperSingleInput` (from `extract_paraformer_feature.py`)
   - Export only the encoder hidden states to ONNX format
   - Save the model as `weights/paraformer_hidden.onnx`

3. **Export Details**

   - **Input**: Frontend features (after LFR) with shape `[batch, time, 560]`
     - Feature dimension: 80 mels × 7 (LFR) = 560
     - Fixed time dimension: 150 frames (inputs longer than this will be truncated)
   - **Output**: Encoder hidden states
   - **Format**: FP32 ONNX model (opset version 17)
   - **Size**: ~603MB

4. **Test the Exported Model**

   Verify the exported ONNX model:
   ```bash
   cd utils
   python test_paraformer_onnx.py
   ```

   This will:
   - Load the ONNX model
   - Display input/output shapes
   - Run inference with dummy data
   - Verify the model works correctly

#### Important Notes

- The exported model uses a **fixed time dimension of 150 frames**
- Input sequences longer than 150 frames will cause runtime errors
- The frontend should truncate or pad inputs to exactly 150 frames
- The wrapper bypasses the frontend FFT to avoid `aten::fft_rfft` operations during export
- If the model is larger than 2GB, weights will be stored in a separate `.onnx_data` file

## Browser Compatibility

- **WebGPU**: Chrome/Edge 113+, Safari 18+ (experimental)
- **WASM Fallback**: All modern browsers (Chrome, Firefox, Safari, Edge)
- **Audio APIs**: Modern browsers with Web Audio API support

## Performance

- **Feature Extraction**: ~1-2 seconds for 5 seconds of audio (depends on device)
- **Video Generation**: ~5-10 seconds for 150 frames (depends on device and WebGPU availability)
- **Memory Usage**: ~1-2GB RAM (mainly for model loading)

## Thanks

This project is based on [HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar), with modifications to enable browser-only execution.

We are grateful for the following open-source projects:

- [LiteAvatar](https://github.com/HumanAIGC/lite-avatar) - Original real-time 2D chat avatar project
- [Paraformer](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) & [FunASR](https://github.com/modelscope/FunASR) - Audio feature extraction
- [HeadTTS](https://github.com/met4citizen/HeadTTS) - Reference implementation
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) - Browser inference engine

## License

MIT License
