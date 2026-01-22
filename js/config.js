const config = {
    cdnBaseUrl: 'https://huggingface.co/fushengji/lite-avatar-wasm/resolve/main',
    
    modelPaths: {
        paraformerFp32: './paraformer_hidden.onnx',
        audio2mouth: './model_1.onnx',
        encoder: './data/preload/net_encode.onnx',
        generator: './data/preload/net_decode.onnx'
    },
    
    chunkedModel: {
        enabled: false,
        chunkSize: 50 * 1024 * 1024,
        totalChunks: 13
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = config;
} else {
    window.appConfig = config;
}

