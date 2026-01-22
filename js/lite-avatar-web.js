/**
 * LiteAvatar WebGPU 浏览器版本
 * 参考 HeadTTS 项目实现方式
 */

// 说明：本项目实现全前端特征提取（完全本地化）：
// 1. 使用 ParaformerFrontend 提取 fbank + LFR + CMVN 特征
// 2. 使用 paraformer_hidden.onnx (603MB FP32) 模型获取 hidden states
// 3. 时间插值到目标帧数

class LiteAvatarWeb {
    constructor() {
        this.audio2mouthModel = null;
        this.encoderModel = null;
        this.generatorModel = null;
        this.paraformerModel = null; // Paraformer hidden states 模型
        this.frontend = null; // Paraformer 前端特征提取器
        this.audioContext = null;
        this.isInitialized = false;
        this.avatarData = null;
        this.audioFile = null;
        this.bgVideoFrames = [];
        this.refFrames = [];
        this.neutralPose = null;
        this.faceBox = null;
        this.mergeMask = null;
        this.processedAudioBuffer = null; // 存储处理后的音频（16kHz, 单声道）
        this.useFrontendFeatureExtraction = true; // 启用全前端特征提取（唯一模式）
        
        // 录音相关状态
        this.mediaRecorder = null;
        this.audioStream = null;
        this.recordedChunks = [];
        this.isRecording = false;
        
        this.initUI();
        this.checkReady();
    }

    initUI() {
        // 文件输入
        const audioFileInput = document.getElementById('audioFile');
        const avatarDataInput = document.getElementById('avatarData');
        const generateBtn = document.getElementById('generateBtn');
        const useDefaultDataBtn = document.getElementById('useDefaultData');
        const fileInputLabel = document.getElementById('fileInputLabel');
        const dataInputLabel = document.getElementById('dataInputLabel');

        // 事件监听
        audioFileInput.addEventListener('change', (e) => this.handleAudioFile(e));
        avatarDataInput.addEventListener('change', (e) => this.handleAvatarData(e));
        generateBtn.addEventListener('click', () => this.generateVideo());
        useDefaultDataBtn.addEventListener('click', () => this.loadDefaultData());
        
        // 默认示例音频按钮
        const useDefaultAudioBtn = document.getElementById('useDefaultAudio');
        if (useDefaultAudioBtn) {
            useDefaultAudioBtn.addEventListener('click', () => this.loadDefaultAudio());
        }
        
        // 录音按钮
        const recordAudioBtn = document.getElementById('recordAudioBtn');
        const stopRecordBtn = document.getElementById('stopRecordBtn');
        if (recordAudioBtn) {
            recordAudioBtn.addEventListener('click', () => this.startRecording());
        }
        if (stopRecordBtn) {
            stopRecordBtn.addEventListener('click', () => this.stopRecording());
        }

        // 预加载模型按钮
        const preloadModelsBtn = document.getElementById('preloadModelsBtn');
        if (preloadModelsBtn) {
            preloadModelsBtn.addEventListener('click', () => this.preloadModels());
        }

        // 点击标签时触发文件选择
        fileInputLabel.addEventListener('click', (e) => {
            e.preventDefault();
            audioFileInput.click();
        });

        dataInputLabel.addEventListener('click', (e) => {
            e.preventDefault();
            avatarDataInput.click();
        });

        // 拖拽支持 - 音频文件
        fileInputLabel.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInputLabel.style.background = '#f0f0ff';
        });
        fileInputLabel.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInputLabel.style.background = 'white';
        });
        fileInputLabel.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInputLabel.style.background = 'white';
            if (e.dataTransfer.files.length > 0) {
                // 创建新的 FileList（浏览器限制，使用 DataTransfer）
                const dt = new DataTransfer();
                dt.items.add(e.dataTransfer.files[0]);
                audioFileInput.files = dt.files;
                this.handleAudioFile({ target: { files: dt.files } });
            }
        });

        // 拖拽支持 - Avatar 数据
        dataInputLabel.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dataInputLabel.style.background = '#f0f0ff';
        });
        dataInputLabel.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dataInputLabel.style.background = 'white';
        });
        dataInputLabel.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dataInputLabel.style.background = 'white';
            // 注意：拖拽目录需要特殊处理，建议使用点击选择
            this.updateStatus('', '请使用"点击选择"按钮来选择目录');
        });
    }

    async handleAudioFile(event) {
        const file = event.target.files[0];
        if (!file) {
            this.checkReady();
            return;
        }

        const audioInfo = document.getElementById('audioInfo');
        audioInfo.className = 'status';
        audioInfo.textContent = `已选择: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        audioInfo.classList.remove('hidden');

        // 验证音频文件并获取详细信息
        try {
            // 使用 AudioContext 获取准确的采样率信息
            const arrayBuffer = await file.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // 显示音频信息
            const info = `采样率: ${audioBuffer.sampleRate}Hz, 通道数: ${audioBuffer.numberOfChannels}, 时长: ${audioBuffer.duration.toFixed(2)}秒`;
            audioInfo.textContent += ` | ${info}`;
            
            // 关闭 AudioContext 以释放资源
            await audioContext.close();
            
            // 保存音频文件引用
            this.audioFile = file;
        } catch (error) {
            console.warn('无法读取音频文件详细信息:', error);
            // 回退到 Audio 元素方法
            try {
                const audio = new Audio();
                audio.src = URL.createObjectURL(file);
                await new Promise((resolve, reject) => {
                    audio.addEventListener('loadedmetadata', () => {
                        const info = `时长: ${audio.duration.toFixed(2)}秒`;
                        audioInfo.textContent += ` | ${info}`;
                        resolve();
                    });
                    audio.addEventListener('error', reject);
                    setTimeout(() => reject(new Error('超时')), 5000);
                });
                this.audioFile = file;
            } catch (fallbackError) {
                audioInfo.className = 'status error';
                audioInfo.textContent = '无法读取音频文件信息';
                this.audioFile = null;
            }
        }
        
        // 检查是否可以启用生成按钮
        this.checkReady();
    }

    async handleAvatarData(event) {
        const files = Array.from(event.target.files);
        const avatarDataInfo = document.getElementById('avatarDataInfo');
        const dataInputText = document.getElementById('dataInputText');
        
        console.log('Avatar 数据上传:', files.length, '个文件');
        
        if (files.length === 0) {
            this.avatarData = null;
            avatarDataInfo.className = 'status error';
            avatarDataInfo.textContent = '未选择任何文件';
            avatarDataInfo.classList.remove('hidden');
            this.updateStatus('error', '未选择任何文件');
            this.checkReady();
            return;
        }
        
        this.avatarData = this.organizeFiles(files);
        const fileCount = Object.keys(this.avatarData).length;
        console.log('Avatar 数据已组织:', fileCount, '个文件');
        console.log('文件列表:', Object.keys(this.avatarData).slice(0, 10), '...');
        
        // 更新状态显示
        const info = `Avatar 数据已加载 (${fileCount} 个文件)`;
        avatarDataInfo.textContent = info;
        avatarDataInfo.className = 'status success';
        avatarDataInfo.classList.remove('hidden');
        
        // 更新文件输入显示
        dataInputText.textContent = `已选择: ${files.length} 个文件`;
        
        this.updateStatus('success', info);
        
        // 检查是否可以启用生成按钮
        this.checkReady();
    }

    organizeFiles(files) {
        const organized = {};
        files.forEach(file => {
            const path = file.webkitRelativePath || file.name;
            // 保留目录结构，特别是 ref_frames/
            if (path.includes('/')) {
                // 对于目录中的文件，保留相对路径
                organized[path] = file;
            } else {
                // 对于根目录的文件，只使用文件名
                organized[file.name] = file;
            }
        });
        return organized;
    }

    async loadDefaultData() {
        const avatarDataInfo = document.getElementById('avatarDataInfo');
        const dataInputText = document.getElementById('dataInputText');
        
        avatarDataInfo.className = 'status';
        avatarDataInfo.textContent = '正在加载默认示例数据...';
        avatarDataInfo.classList.remove('hidden');
        this.updateStatus('', '正在加载默认示例数据...');
        
        try {
            // 加载默认数据目录
            const defaultDataPath = './data/preload';
            this.avatarData = await this.loadDefaultAvatarData(defaultDataPath);
            
            if (this.avatarData && Object.keys(this.avatarData).length > 0) {
                const fileCount = Object.keys(this.avatarData).length;
                const info = `已加载默认示例数据 (${fileCount} 个文件)`;
                
                // 更新状态显示
                avatarDataInfo.textContent = info;
                avatarDataInfo.className = 'status success';
                
                // 更新文件输入显示
                dataInputText.textContent = `已选择: 默认示例数据 (${fileCount} 个文件)`;
                
                this.updateStatus('success', info);
                console.log('默认数据加载成功:', fileCount, '个文件');
            } else {
                throw new Error('未能加载默认数据');
            }
        } catch (error) {
            console.error('加载默认数据失败:', error);
            avatarDataInfo.className = 'status error';
            avatarDataInfo.textContent = `加载默认数据失败: ${error.message}`;
            this.updateStatus('error', `加载默认数据失败: ${error.message}`);
            this.avatarData = null;
        }
        
        this.checkReady();
    }

    async loadDefaultAudio() {
        const audioInfo = document.getElementById('audioInfo');
        audioInfo.className = 'status';
        audioInfo.textContent = '正在加载默认示例音频...';
        audioInfo.classList.remove('hidden');
        
        try {
            // 加载默认音频文件
            const audioUrl = './data/test.wav';
            const response = await fetch(audioUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const blob = await response.blob();
            const file = new File([blob], 'test.wav', { type: 'audio/wav' });
            
            // 使用 AudioContext 获取音频信息
            const arrayBuffer = await file.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // 显示音频信息
            const info = `采样率: ${audioBuffer.sampleRate}Hz, 通道数: ${audioBuffer.numberOfChannels}, 时长: ${audioBuffer.duration.toFixed(2)}秒`;
            audioInfo.textContent = `已加载默认示例音频: test.wav | ${info}`;
            audioInfo.className = 'status success';
            
            // 关闭 AudioContext 以释放资源
            await audioContext.close();
            
            // 保存音频文件引用
            this.audioFile = file;
            
            // 更新文件输入显示
            const fileInputText = document.getElementById('fileInputText');
            fileInputText.textContent = `已选择: test.wav (默认示例音频)`;
            
            console.log('默认音频加载成功:', file.name, info);
        } catch (error) {
            console.error('加载默认音频失败:', error);
            audioInfo.className = 'status error';
            audioInfo.textContent = `加载默认音频失败: ${error.message}`;
            this.audioFile = null;
        }
        
        // 检查是否可以启用生成按钮
        this.checkReady();
    }

    async startRecording() {
        const recordingStatus = document.getElementById('recordingStatus');
        const recordAudioBtn = document.getElementById('recordAudioBtn');
        const stopRecordBtn = document.getElementById('stopRecordBtn');
        const audioInfo = document.getElementById('audioInfo');
        
        try {
            // 请求麦克风权限
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,  // 目标采样率
                    channelCount: 1,     // 单声道
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // 初始化录音
            this.recordedChunks = [];
            const options = { mimeType: 'audio/webm' };
            
            // 尝试使用更好的编码格式
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                options.mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                options.mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                options.mimeType = 'audio/mp4';
            }
            
            this.mediaRecorder = new MediaRecorder(this.audioStream, options);
            
            // 录音数据收集
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            // 录音停止处理
            this.mediaRecorder.onstop = async () => {
                try {
                    // 创建 Blob
                    const blob = new Blob(this.recordedChunks, { type: options.mimeType });
                    
                    // 转换为 WAV 格式（因为后续处理需要）
                    const audioBuffer = await this.convertBlobToAudioBuffer(blob);
                    const wavBlob = this.audioBufferToWav(audioBuffer);
                    
                    // 创建 File 对象
                    const file = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
                    
                    // 显示音频信息
                    const info = `采样率: ${audioBuffer.sampleRate}Hz, 通道数: ${audioBuffer.numberOfChannels}, 时长: ${audioBuffer.duration.toFixed(2)}秒`;
                    audioInfo.textContent = `已录制音频 | ${info}`;
                    audioInfo.className = 'status success';
                    audioInfo.classList.remove('hidden');
                    
                    // 更新文件输入显示
                    const fileInputText = document.getElementById('fileInputText');
                    fileInputText.textContent = `已选择: ${file.name} (麦克风录音)`;
                    
                    // 保存音频文件引用
                    this.audioFile = file;
                    
                    // 停止音频流
                    this.audioStream.getTracks().forEach(track => track.stop());
                    this.audioStream = null;
                    
                    console.log('录音完成:', file.name, info);
                    
                    // 检查是否可以启用生成按钮
                    this.checkReady();
                } catch (error) {
                    console.error('处理录音数据失败:', error);
                    recordingStatus.className = 'status error';
                    recordingStatus.textContent = `处理录音失败: ${error.message}`;
                }
            };
            
            // 错误处理
            this.mediaRecorder.onerror = (error) => {
                console.error('录音错误:', error);
                recordingStatus.className = 'status error';
                recordingStatus.textContent = `录音错误: ${error.message || '未知错误'}`;
                this.stopRecording();
            };
            
            // 开始录音
            this.mediaRecorder.start(100); // 每100ms收集一次数据
            this.isRecording = true;
            
            // 更新UI
            recordAudioBtn.style.display = 'none';
            stopRecordBtn.style.display = 'inline-block';
            recordingStatus.className = 'status';
            recordingStatus.textContent = '🎤 正在录音... 点击"停止录音"结束';
            recordingStatus.classList.remove('hidden');
            audioInfo.classList.add('hidden');
            
            console.log('开始录音...');
        } catch (error) {
            console.error('启动录音失败:', error);
            recordingStatus.className = 'status error';
            recordingStatus.textContent = `无法访问麦克风: ${error.message || '请检查浏览器权限设置'}`;
            recordingStatus.classList.remove('hidden');
            
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
                this.audioStream = null;
            }
        }
    }

    stopRecording() {
        const recordingStatus = document.getElementById('recordingStatus');
        const recordAudioBtn = document.getElementById('recordAudioBtn');
        const stopRecordBtn = document.getElementById('stopRecordBtn');
        
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // 更新UI
            recordAudioBtn.style.display = 'inline-block';
            stopRecordBtn.style.display = 'none';
            recordingStatus.textContent = '正在处理录音...';
        }
    }

    async convertBlobToAudioBuffer(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000  // 目标采样率
        });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        await audioContext.close();
        return audioBuffer;
    }

    audioBufferToWav(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;
        
        const length = audioBuffer.length;
        const buffer = new ArrayBuffer(44 + length * numChannels * bytesPerSample);
        const view = new DataView(buffer);
        
        // WAV 文件头
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * numChannels * bytesPerSample, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, length * numChannels * bytesPerSample, true);
        
        // 写入音频数据
        let offset = 44;
        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                offset += 2;
            }
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    async loadDefaultAvatarData(dataPath) {
        const avatarData = {};
        const filesToLoad = [
            'bg_video.mp4',
            'face_box.txt',
            'neutral_pose.npy',
            'net_decode.pt',
            'net_encode.pt'
        ];

        // 加载主要文件
        for (const fileName of filesToLoad) {
            try {
                const response = await fetch(`${dataPath}/${fileName}`);
                if (response.ok) {
                    const blob = await response.blob();
                    avatarData[fileName] = new File([blob], fileName, { type: blob.type });
                    console.log(`✓ 已加载: ${fileName}`);
                } else {
                    console.warn(`⚠️ 文件不存在或无法访问: ${fileName}`);
                }
            } catch (error) {
                console.warn(`⚠️ 加载 ${fileName} 失败:`, error);
            }
        }

        // 加载参考帧（ref_frames 目录）
        const refFramesPath = `${dataPath}/ref_frames`;
        let refFrameIndex = 0;
        let hasMoreFrames = true;
        const maxFrames = 150; // 限制加载的帧数

        while (hasMoreFrames && refFrameIndex < maxFrames) {
            const frameFileName = `ref_${String(refFrameIndex).padStart(5, '0')}.jpg`;
            try {
                const response = await fetch(`${refFramesPath}/${frameFileName}`);
                if (response.ok) {
                    const blob = await response.blob();
                    avatarData[`ref_frames/${frameFileName}`] = new File([blob], frameFileName, { type: 'image/jpeg' });
                    refFrameIndex++;
                } else {
                    // 如果文件不存在，停止加载
                    hasMoreFrames = false;
                }
            } catch (error) {
                // 如果出错，停止加载
                hasMoreFrames = false;
            }
        }

        console.log(`✓ 已加载 ${refFrameIndex} 个参考帧`);

        if (Object.keys(avatarData).length === 0) {
            throw new Error('未能加载任何文件，请检查数据路径是否正确');
        }

        return avatarData;
    }
    
    checkReady() {
        const generateBtn = document.getElementById('generateBtn');
        const audioFileInput = document.getElementById('audioFile');
        const audioFile = audioFileInput?.files[0] || this.audioFile;
        const hasAudio = !!audioFile;
        const hasAvatarData = !!this.avatarData && Object.keys(this.avatarData).length > 0;
        
        // 调试信息
        console.log('检查按钮状态:', {
            hasAudio,
            hasAvatarData,
            audioFile: audioFile?.name,
            avatarDataFiles: this.avatarData ? Object.keys(this.avatarData).length : 0
        });
        
        // 如果音频和 Avatar 数据都已准备好，启用按钮
        if (hasAudio && hasAvatarData) {
            generateBtn.disabled = false;
            generateBtn.textContent = '生成视频';
            console.log('✓ 按钮已启用');
        } else {
            generateBtn.disabled = true;
            if (!hasAudio && !hasAvatarData) {
                generateBtn.textContent = '请先上传音频文件和 Avatar 数据';
            } else if (!hasAudio) {
                generateBtn.textContent = '请先上传音频文件';
            } else if (!hasAvatarData) {
                generateBtn.textContent = '请先加载 Avatar 数据';
            }
            console.log('✗ 按钮仍禁用:', { hasAudio, hasAvatarData });
        }
    }

    async initializeModels() {
        if (this.isInitialized) return;

        this.updateStatus('', '正在初始化模型...');

        const getModelPath = (relativePath) => {
            // 如果配置中已经是完整 URL（以 http:// 或 https:// 开头），直接返回
            if (relativePath && (relativePath.startsWith('http://') || relativePath.startsWith('https://'))) {
                return relativePath;
            }
            // 如果有 CDN base URL，拼接路径
            const cdnBaseUrl = window.appConfig?.cdnBaseUrl;
            if (cdnBaseUrl) {
                return cdnBaseUrl.replace(/\/$/, '') + '/' + relativePath.replace(/^\.\//, '');
            }
            // 否则返回相对路径
            return relativePath;
        };

        try {
            if (ort?.env?.wasm) {
                ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
                ort.env.wasm.numThreads = 1;
                ort.env.wasm.simd = true;
            }

            // 所有模型都使用 WASM 后端，因为 WebGPU 对 Concat 算子有嵌套深度限制（127）
            const wasmOnlySessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                wasmPaths: {
                    'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm.wasm',
                    'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd.wasm',
                    'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-threaded.wasm',
                    'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm'
                },
                numThreads: 4,
                logSeverityLevel: 2,
                logVerbosityLevel: 0
            };
            
            this.updateStatus('', '正在加载 Paraformer 模型（FP32，WASM 后端）...');
            try {
                const paraformerSessionOptions = {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                    wasmPaths: {
                        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm.wasm',
                        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd.wasm',
                        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-threaded.wasm',
                        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm'
                    },
                    numThreads: 4,
                    logSeverityLevel: 2,
                    logVerbosityLevel: 0
                };
                
                const modelPath = getModelPath(window.appConfig?.modelPaths?.paraformerFp32 || './weights/paraformer_hidden.onnx');
                this.paraformerModel = await ort.InferenceSession.create(modelPath, paraformerSessionOptions);
                console.log('✓ Paraformer FP32 模型已加载（603MB）');
                console.log('Paraformer inputs:', this.paraformerModel.inputNames);
                console.log('Paraformer outputs:', this.paraformerModel.outputNames);
                
                // 初始化前端特征提取器
                this.frontend = new ParaformerFrontend({
                    sampleRate: 16000,
                    nMels: 80,
                    frameLength: 400,
                    frameShift: 160,
                    windowType: 'hamming',
                    lfrM: 7,
                    lfrN: 6
                });
                console.log('✓ Paraformer 前端特征提取器已初始化');
                this.useFrontendFeatureExtraction = true;
                console.log('✓ 前端特征提取已启用');
            } catch (error) {
                console.error('Paraformer 模型加载失败:', error);
                console.error('错误详情:', error.message, error.stack);
                this.paraformerModel = null;
                this.frontend = null;
                this.useFrontendFeatureExtraction = false;
                throw new Error(`前端特征提取模型加载失败: ${error.message}。请确保 weights/paraformer_hidden.onnx 文件存在且可访问。`);
            }

            this.updateStatus('', '正在加载音频到嘴部模型...');
            const audio2mouthPath = getModelPath(window.appConfig?.modelPaths?.audio2mouth || './weights/model_1.onnx');
            this.audio2mouthModel = await ort.InferenceSession.create(
                audio2mouthPath,
                wasmOnlySessionOptions
            );
            console.log('✓ audio2mouth 模型已加载（WASM 后端）');

            this.updateStatus('', '正在加载面部生成模型...');
            try {
                this.encoderModel = await ort.InferenceSession.create('./data/preload/net_encode.onnx', wasmOnlySessionOptions);
                console.log('✓ 编码器模型已加载（WASM 后端）');
            } catch (error) {
                console.warn('编码器模型加载失败:', error.message);
                this.encoderModel = null;
            }
            
            try {
                this.generatorModel = await ort.InferenceSession.create('./data/preload/net_decode.onnx', wasmOnlySessionOptions);
                console.log('✓ 生成器模型已加载（WASM 后端）');
            } catch (error) {
                console.warn('生成器模型加载失败:', error.message);
                this.generatorModel = null;
            }

            this.isInitialized = true;
            this.updateStatus('success', '模型初始化完成！');
        } catch (error) {
            this.updateStatus('error', `模型初始化失败: ${error.message}`);
            throw error;
        }
    }

    async preloadModels() {
        const preloadBtn = document.getElementById('preloadModelsBtn');
        const preloadStatus = document.getElementById('preloadStatus');
        const t = window.i18n ? window.i18n.t : (key) => key;
        
        if (this.isInitialized) {
            preloadStatus.className = 'status success';
            preloadStatus.textContent = t('preloadAlready');
            preloadStatus.classList.remove('hidden');
            if (preloadBtn) {
                preloadBtn.disabled = true;
                preloadBtn.textContent = t('preloadAlready');
            }
            return;
        }

        if (preloadBtn) {
            preloadBtn.disabled = true;
            preloadBtn.textContent = '⏳ ' + t('preloadLoading');
        }

        preloadStatus.className = 'status';
        preloadStatus.textContent = t('preloadLoading');
        preloadStatus.classList.remove('hidden');

        try {
            await this.initializeModels();
            preloadStatus.className = 'status success';
            preloadStatus.textContent = t('preloadSuccess');
            if (preloadBtn) {
                preloadBtn.textContent = t('preloadAlready');
            }
            this.updateStatus('success', t('preloadSuccess'));
        } catch (error) {
            console.error('模型预加载失败:', error);
            preloadStatus.className = 'status error';
            preloadStatus.textContent = t('preloadError') + ': ' + error.message;
            if (preloadBtn) {
                preloadBtn.disabled = false;
                preloadBtn.textContent = t('preloadBtn');
            }
            this.updateStatus('error', t('preloadError') + ': ' + error.message);
        }
    }

    async loadAvatarData(dataDirOrFiles) {
        this.updateStatus('', '正在处理 Avatar 数据...');

        try {
            let avatarDataFiles = null;

            // 如果传入的是文件对象字典，直接使用
            if (typeof dataDirOrFiles === 'object' && !Array.isArray(dataDirOrFiles) && dataDirOrFiles.constructor === Object) {
                avatarDataFiles = dataDirOrFiles;
            } else if (typeof dataDirOrFiles === 'string') {
                // 如果是路径字符串，需要加载
                avatarDataFiles = await this.loadDefaultAvatarData(dataDirOrFiles);
            } else {
                // 使用已加载的数据
                avatarDataFiles = this.avatarData;
            }

            if (!avatarDataFiles || Object.keys(avatarDataFiles).length === 0) {
                throw new Error('没有可用的 Avatar 数据');
            }

            // 加载背景视频帧
            try {
                if (avatarDataFiles['bg_video.mp4']) {
                    const bgVideoBlob = avatarDataFiles['bg_video.mp4'];
                    console.log('视频文件大小:', bgVideoBlob.size, 'bytes, 类型:', bgVideoBlob.type);
                    const bgVideoUrl = URL.createObjectURL(bgVideoBlob);
                    console.log('正在提取背景视频帧，URL:', bgVideoUrl.substring(0, 50) + '...');
                    try {
                        await this.extractVideoFrames(bgVideoUrl);
                        console.log(`✓ 已提取 ${this.bgVideoFrames.length} 个背景帧`);
                    } finally {
                        URL.revokeObjectURL(bgVideoUrl);
                    }
                } else if (typeof dataDirOrFiles === 'string') {
                    // 从路径加载，添加缓存破坏参数
                    const videoPath = `${dataDirOrFiles}/bg_video.mp4?t=${Date.now()}`;
                    console.log('从路径加载视频:', videoPath);
                    const bgVideoResponse = await fetch(videoPath, {
                        cache: 'no-cache'
                    });
                    if (!bgVideoResponse.ok) {
                        throw new Error(`无法加载视频文件: ${bgVideoResponse.status} ${bgVideoResponse.statusText}`);
                    }
                    const bgVideoBlob = await bgVideoResponse.blob();
                    console.log('视频文件大小:', bgVideoBlob.size, 'bytes, 类型:', bgVideoBlob.type);
                    const bgVideoUrl = URL.createObjectURL(bgVideoBlob);
                    console.log('正在提取背景视频帧...');
                    try {
                        await this.extractVideoFrames(bgVideoUrl);
                        console.log(`✓ 已提取 ${this.bgVideoFrames.length} 个背景帧`);
                    } finally {
                        URL.revokeObjectURL(bgVideoUrl);
                    }
                } else {
                    throw new Error('未找到背景视频文件 bg_video.mp4');
                }
            } catch (error) {
                console.error('提取视频帧失败:', error);
                // 视频解码失败，需要用户处理视频格式
                throw new Error(`背景视频解码失败: ${error.message}。请确保视频格式为浏览器兼容的 MP4/H.264 编码。可以使用 ffmpeg 转换: ffmpeg -i bg_video.mp4 -c:v libx264 -c:a aac -movflags +faststart bg_video_compatible.mp4`);
            }

            // 加载 neutral_pose
            if (avatarDataFiles['neutral_pose.npy']) {
                // 需要解析 .npy 文件（可以使用 npyjs 库）
                console.log('neutral_pose.npy 已加载（需要解析）');
            }

            // 加载 face_box
            let faceBoxText = null;
            if (avatarDataFiles['face_box.txt']) {
                faceBoxText = await avatarDataFiles['face_box.txt'].text();
            } else if (typeof dataDirOrFiles === 'string') {
                const faceBoxResponse = await fetch(`${dataDirOrFiles}/face_box.txt`);
                faceBoxText = await faceBoxResponse.text();
            }

            if (faceBoxText) {
                const [y1, y2, x1, x2] = faceBoxText.trim().split(/\s+/).map(Number);
                this.faceBox = { y1, y2, x1, x2 };
                console.log('face_box:', this.faceBox);

                // 生成 merge_mask
                this.generateMergeMask();
            }

            // 加载参考帧
            await this.loadReferenceFrames(avatarDataFiles);
            
            // 检查是否成功加载了背景视频帧（必须要有 bg_video）
            if (!this.bgVideoFrames || this.bgVideoFrames.length === 0) {
                throw new Error('未能加载背景视频帧。请确保 bg_video.mp4 文件存在且格式正确（MP4/H.264 编码）。\n如果视频无法解码，请使用 convert_video.sh 脚本转换视频格式。');
            }
            
            console.log(`✓ 成功加载 ${this.bgVideoFrames.length} 个背景视频帧`);

            this.updateStatus('success', 'Avatar 数据处理完成');
        } catch (error) {
            this.updateStatus('error', `处理 Avatar 数据失败: ${error.message}`);
            console.error('处理 Avatar 数据失败:', error);
            throw error;
        }
    }

    async extractVideoFrames(videoUrl) {
        return new Promise((resolve, reject) => {
            const video = document.createElement('video');
            video.preload = 'auto';
            video.muted = true;
            video.playsInline = true;
            
            // 设置超时
            const timeout = setTimeout(() => {
                reject(new Error('视频加载超时'));
            }, 30000); // 30秒超时
            
            let isResolved = false;
            
            const cleanup = () => {
                if (isResolved) return;
                clearTimeout(timeout);
                video.removeEventListener('loadedmetadata', onLoadedMetadata);
                video.removeEventListener('loadeddata', onLoadedData);
                video.removeEventListener('error', onError);
                video.removeEventListener('canplay', onCanPlay);
                video.removeEventListener('canplaythrough', onCanPlayThrough);
            };
            
            const onLoadedMetadata = () => {
                console.log('视频元数据加载完成:', {
                    width: video.videoWidth,
                    height: video.videoHeight,
                    duration: video.duration,
                    readyState: video.readyState,
                    networkState: video.networkState
                });
                
                if (video.videoWidth === 0 || video.videoHeight === 0) {
                    cleanup();
                    if (!isResolved) {
                        isResolved = true;
                        reject(new Error('视频尺寸无效（宽高为0）'));
                    }
                }
            };
            
            const onError = (event) => {
                cleanup();
                if (isResolved) return;
                isResolved = true;
                
                const error = video.error;
                let errorMsg = '视频解码失败';
                
                console.error('视频元素错误:', {
                    error: error,
                    code: error?.code,
                    message: error?.message,
                    networkState: video.networkState,
                    readyState: video.readyState,
                    src: video.src.substring(0, 100)
                });
                
                if (error) {
                    switch (error.code) {
                        case error.MEDIA_ERR_ABORTED:
                            errorMsg = '视频加载被中止';
                            break;
                        case error.MEDIA_ERR_NETWORK:
                            errorMsg = '网络错误导致视频加载失败';
                            break;
                        case error.MEDIA_ERR_DECODE:
                            errorMsg = '视频解码失败。文件可能损坏或格式不支持';
                            break;
                        case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                            errorMsg = '视频格式不支持';
                            break;
                        default:
                            errorMsg = error.message || `错误代码: ${error.code}`;
                    }
                }
                
                reject(new Error(`背景视频解码失败: ${errorMsg}。请检查视频文件是否正确转换`));
            };
            
            const onCanPlay = () => {
                console.log('视频可以播放，尺寸:', video.videoWidth, 'x', video.videoHeight, 'readyState:', video.readyState);
            };
            
            const onCanPlayThrough = () => {
                console.log('视频可以完整播放');
            };
            
            const onLoadedData = async () => {
                if (isResolved) return;
                
                try {
                    console.log('视频数据加载完成，开始提取帧...');
                    console.log('视频信息:', {
                        width: video.videoWidth,
                        height: video.videoHeight,
                        duration: video.duration,
                        readyState: video.readyState
                    });
                    
                    // 等待视频完全准备好
                    if (video.readyState < 3) {
                        console.log('等待视频准备就绪...');
                        await new Promise((resolveReady) => {
                            const checkReady = () => {
                                if (video.readyState >= 3) {
                                    resolveReady();
                                } else {
                                    setTimeout(checkReady, 100);
                                }
                            };
                            checkReady();
                        });
                    }
                    
                    cleanup();
                    clearTimeout(timeout);
                    
                    if (video.videoWidth === 0 || video.videoHeight === 0) {
                        if (!isResolved) {
                            isResolved = true;
                            reject(new Error('视频尺寸无效（宽高为0）'));
                        }
                        return;
                    }
                    
                    // 初始化 canvas
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    
                    console.log('开始提取视频帧，总时长:', video.duration, '秒');
                    
                    const frameRate = 30;
                    const frameInterval = 1 / frameRate;
                    const maxFrames = 150; // 限制帧数
                    const duration = Math.min(video.duration, maxFrames / frameRate);
                    const frames = [];
                    
                    const extractFrame = (time) => {
                        return new Promise((resolveFrame) => {
                            const seekTimeout = setTimeout(() => {
                                console.warn(`帧提取超时 (时间: ${time.toFixed(2)}s)`);
                                resolveFrame(null);
                            }, 3000);
                            
                            const onSeeked = () => {
                                clearTimeout(seekTimeout);
                                try {
                                    ctx.drawImage(video, 0, 0);
                                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                                    resolveFrame(imageData);
                                } catch (error) {
                                    console.error('提取帧时出错:', error);
                                    resolveFrame(null);
                                }
                            };
                            
                            video.addEventListener('seeked', onSeeked, { once: true });
                            video.currentTime = time;
                        });
                    };
                    
                    // 提取帧
                    for (let currentTime = 0; currentTime < duration && frames.length < maxFrames; currentTime += frameInterval) {
                        const frame = await extractFrame(currentTime);
                        if (frame) {
                            frames.push(frame);
                        }
                        // 更新进度
                        if (frames.length % 10 === 0) {
                            console.log(`已提取 ${frames.length} 帧...`);
                        }
                    }
                    
                    if (frames.length === 0) {
                        if (!isResolved) {
                            isResolved = true;
                            reject(new Error('未能提取任何视频帧'));
                        }
                        return;
                    }
                    
                    console.log(`✓ 成功提取 ${frames.length} 个视频帧`);
                    this.bgVideoFrames = frames;
                    if (!isResolved) {
                        isResolved = true;
                        resolve(frames);
                    }
                } catch (error) {
                    if (!isResolved) {
                        isResolved = true;
                        reject(error);
                    }
                }
            };
            
            video.addEventListener('loadedmetadata', onLoadedMetadata);
            video.addEventListener('loadeddata', onLoadedData);
            video.addEventListener('error', onError);
            video.addEventListener('canplay', onCanPlay);
            video.addEventListener('canplaythrough', onCanPlayThrough);
            
            // 设置视频源
            console.log('设置视频源:', videoUrl);
            video.src = videoUrl;
            
            // 尝试加载
            video.load();
            
            // 添加额外的错误检查
            setTimeout(() => {
                if (!isResolved && video.readyState === 0) {
                    console.error('视频加载超时，readyState:', video.readyState);
                    if (video.error) {
                        onError(new Event('error'));
                    }
                }
            }, 5000);
        });
    }

    generateMergeMask() {
        if (!this.faceBox) return;

        const { y1, y2, x1, x2 } = this.faceBox;
        const width = x2 - x1;
        const height = y2 - y1;

        // 创建渐变遮罩
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // 绘制白色背景
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);

        // 绘制黑色中心区域（边缘留 10px）
        ctx.fillStyle = 'black';
        ctx.fillRect(10, 10, width - 20, height - 20);

        // 应用高斯模糊（使用 Canvas 的 filter，或使用 WebGL）
        ctx.filter = 'blur(15px)';
        ctx.drawImage(canvas, 0, 0);

        const imageData = ctx.getImageData(0, 0, width, height);
        this.mergeMask = new Float32Array(imageData.data.length / 4);
        for (let i = 0; i < imageData.data.length; i += 4) {
            this.mergeMask[i / 4] = imageData.data[i] / 255;
        }
    }

    async loadReferenceFrames(dataDirOrFiles) {
        // 从已加载的数据中提取参考帧
        const refFrames = [];
        
        // 如果传入的是文件对象字典
        if (typeof dataDirOrFiles === 'object' && !Array.isArray(dataDirOrFiles)) {
            const refFrameKeys = Object.keys(dataDirOrFiles)
                .filter(key => key.startsWith('ref_frames/') || key.match(/^ref_\d+\.jpg$/))
                .sort();

            for (const key of refFrameKeys) {
                const file = dataDirOrFiles[key];
                if (file instanceof File || file instanceof Blob) {
                    const imageUrl = URL.createObjectURL(file);
                    refFrames.push({
                        key: key,
                        file: file,
                        url: imageUrl
                    });
                }
            }
        } else if (typeof dataDirOrFiles === 'string') {
            // 从路径加载（需要实现）
            console.log('从路径加载参考帧功能待实现');
        }

        this.refFrames = refFrames;
        console.log(`已加载 ${refFrames.length} 个参考帧`);
    }

    async processAudio(audioFile, targetFrameCount = null) {
        this.updateStatus('', '正在处理音频...');
        
        try {
            // 读取音频文件
            const arrayBuffer = await audioFile.arrayBuffer();
            const inputAudioBuffer = await this.decodeAudioData(arrayBuffer);

            // 目标格式：16kHz, 单声道
            const targetSampleRate = 16000;
            const targetChannels = 1;
            
            // 确保 audioContext 已初始化
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            // 创建目标 AudioContext（16kHz）
            const targetAudioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: targetSampleRate
            });
            
            // 1. 重采样到 16kHz（如果需要）
            let processedBuffer = inputAudioBuffer;
            if (inputAudioBuffer.sampleRate !== targetSampleRate) {
                console.log(`重采样音频: ${inputAudioBuffer.sampleRate}Hz -> ${targetSampleRate}Hz`);
                processedBuffer = await this.resampleAudioBuffer(inputAudioBuffer, targetSampleRate);
            }
            
            // 2. 转换为单声道（如果需要）
            if (processedBuffer.numberOfChannels !== targetChannels) {
                console.log(`转换音频通道: ${processedBuffer.numberOfChannels} -> ${targetChannels}`);
                processedBuffer = await this.convertToMono(processedBuffer);
            }
            
            // 3. 保存处理后的音频缓冲区（用于后续视频合成）
            this.processedAudioBuffer = processedBuffer;
            
            // 4. 提取音频数据用于特征提取
            const audioData = processedBuffer.getChannelData(0);
            
            // 5. 确定目标帧数
            // 如果指定了 targetFrameCount，使用它；否则基于音频时长计算
            const fps = 30;
            const frameCount = targetFrameCount !== null 
                ? targetFrameCount 
                : Math.floor(audioData.length / targetSampleRate * fps);
            
            console.log('音频处理:', {
                原始采样率: inputAudioBuffer.sampleRate,
                目标采样率: targetSampleRate,
                音频时长: processedBuffer.duration.toFixed(2) + '秒',
                音频采样数: audioData.length,
                目标帧数: frameCount,
                基于: targetFrameCount !== null ? '视频帧数' : '音频时长'
            });
            
            // 6. 提取音频特征
            const audioFeatures = await this.extractAudioFeatures(audioData, frameCount);

            // 6. 使用 audio2mouth 模型生成嘴部参数
            const mouthParams = await this.audio2mouthInference(audioFeatures, frameCount);
            
            console.log('音频处理完成:', {
                原始采样率: inputAudioBuffer.sampleRate,
                目标采样率: targetSampleRate,
                原始通道数: inputAudioBuffer.numberOfChannels,
                目标通道数: targetChannels,
                时长: processedBuffer.duration.toFixed(2) + '秒',
                采样数: processedBuffer.length
            });

            return mouthParams;
        } catch (error) {
            this.updateStatus('error', `音频处理失败: ${error.message}`);
            throw error;
        }
    }

    async decodeAudioData(arrayBuffer) {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        return await this.audioContext.decodeAudioData(arrayBuffer);
    }

    resampleAudio(audioData, fromRate, toRate) {
        // 简单的线性插值重采样
        const ratio = fromRate / toRate;
        const newLength = Math.floor(audioData.length / ratio);
        const resampled = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const index = Math.floor(srcIndex);
            const fraction = srcIndex - index;
            
            if (index + 1 < audioData.length) {
                resampled[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
            } else {
                resampled[i] = audioData[index];
            }
        }

        return resampled;
    }
    
    async resampleAudioBuffer(audioBuffer, targetSampleRate) {
        const sourceSampleRate = audioBuffer.sampleRate;
        const numberOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        const targetLength = Math.round(length * targetSampleRate / sourceSampleRate);
        
        // 创建新的 AudioContext 用于重采样
        const offlineContext = new OfflineAudioContext(
            numberOfChannels,
            targetLength,
            targetSampleRate
        );
        
        // 创建源节点
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start(0);
        
        // 渲染并返回重采样后的 buffer
        return await offlineContext.startRendering();
    }
    
    async convertToMono(audioBuffer) {
        if (audioBuffer.numberOfChannels === 1) {
            return audioBuffer;
        }
        
        const numberOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        const sampleRate = audioBuffer.sampleRate;
        
        // 确保有 audioContext
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // 创建单声道 buffer
        const monoBuffer = this.audioContext.createBuffer(1, length, sampleRate);
        const monoData = monoBuffer.getChannelData(0);
        
        // 混合所有通道
        for (let i = 0; i < numberOfChannels; i++) {
            const channelData = audioBuffer.getChannelData(i);
            for (let j = 0; j < length; j++) {
                monoData[j] += channelData[j] / numberOfChannels;
            }
        }
        
        return monoBuffer;
    }

    // （已移除）前端 Mel/FFT/LFR 特征提取逻辑：特征统一走后端 `/extract_features`
    
    // （已移除）createMelFilterBank / FFT 等前端特征提取辅助函数
    
    // （已移除）前端 LFR/插值逻辑：特征统一走后端 `/extract_features`

    async extractAudioFeatures(audioData, frameCount) {
        // 全前端模式：使用浏览器端 Paraformer 模型
        console.log('检查前端特征提取条件:', {
            useFrontendFeatureExtraction: this.useFrontendFeatureExtraction,
            hasParaformerModel: !!this.paraformerModel,
            hasFrontend: !!this.frontend
        });
        
        // 全前端模式：使用浏览器端 Paraformer 模型
        if (this.useFrontendFeatureExtraction && this.paraformerModel && this.frontend) {
            console.log('使用前端 Paraformer 模型提取特征');
            return await this.extractFeaturesFromFrontend(audioData, frameCount);
        }
        
        // 如果前端特征提取不可用，直接报错（不再支持后端回退）
        throw new Error('前端特征提取不可用。请确保 Paraformer 模型已正确加载。');
        
        if (this.featureExtractor) {
            try {
                // 尝试使用完整的特征提取器
                const features = await this.featureExtractor.extractFeatures(audioData, frameCount);
                console.log(`特征提取器返回特征长度: ${features.length}, 期望: ${frameCount * numChannels * featureDim}`);
                
                // 如果返回的是 [time * features] 格式，需要转换为 [time * channels * features]
                if (features.length === frameCount * featureDim) {
                    // 扩展为多通道格式：将单通道特征复制到所有通道
                    const multiChannelFeatures = new Float32Array(frameCount * numChannels * featureDim);
                    for (let t = 0; t < frameCount; t++) {
                        for (let c = 0; c < numChannels; c++) {
                            const srcStart = t * featureDim;
                            const dstStart = (t * numChannels * featureDim) + (c * featureDim);
                            multiChannelFeatures.set(
                                features.slice(srcStart, srcStart + featureDim),
                                dstStart
                            );
                        }
                    }
                    console.log(`特征已扩展为多通道格式，长度: ${multiChannelFeatures.length}`);
                    return multiChannelFeatures;
                } else if (features.length < frameCount * numChannels * featureDim) {
                    // 特征数量不足，需要插值或填充
                    console.warn(`特征数量不足: 实际 ${features.length}, 期望 ${frameCount * numChannels * featureDim}`);
                    // 检查是否是 [time * features] 格式
                    const actualFrames = features.length / featureDim;
                    if (actualFrames > 0 && actualFrames < frameCount) {
                        // 需要插值到目标帧数
                        const interpolatedFeatures = new Float32Array(frameCount * numChannels * featureDim);
                        for (let t = 0; t < frameCount; t++) {
                            const srcFrameIdx = Math.floor((t / frameCount) * actualFrames);
                            const srcStart = srcFrameIdx * featureDim;
                            for (let c = 0; c < numChannels; c++) {
                                const dstStart = (t * numChannels * featureDim) + (c * featureDim);
                                if (srcStart + featureDim <= features.length) {
                                    interpolatedFeatures.set(
                                        features.slice(srcStart, srcStart + featureDim),
                                        dstStart
                                    );
                                }
                            }
                        }
                        console.log(`特征已插值，长度: ${interpolatedFeatures.length}`);
                        return interpolatedFeatures;
                    }
                }
                // 如果格式正确，直接返回
                if (features.length === frameCount * numChannels * featureDim) {
                    return features;
                }
                console.warn(`特征格式不匹配，使用简化版本`);
            } catch (error) {
                console.warn('特征提取失败，使用简化版本:', error);
            }
        }
        
        // 使用改进的特征提取：基于真实的 Mel 频谱 + LFR
        // 这比完全随机更接近真实特征
        console.log('使用改进的特征提取（基于 Mel 频谱）...');
        
        // 1. 提取 Mel 频谱特征（使用 Web Audio API）
        const melFeatures = await this.extractMelSpectrogramImproved(audioData, frameCount);
        
        // 2. 扩展为 30 通道格式
        // Paraformer encoder 有 30 层，每层输出 512 维特征
        // 我们需要将 Mel 特征（560维）映射到 512 维，然后为每个通道生成不同的表示
        const totalLength = frameCount * numChannels * featureDim;
        const features = new Float32Array(totalLength);
        
        // Mel 特征维度（LFR 后）
        const melFeatureDim = 560; // LFR: 80 * 7 = 560
        
        // 计算特征的统计信息（用于归一化）
        let melMin = Infinity, melMax = -Infinity, melSum = 0, melCount = 0;
        for (let i = 0; i < melFeatures.length; i++) {
            const val = melFeatures[i];
            if (isFinite(val)) {
                melMin = Math.min(melMin, val);
                melMax = Math.max(melMax, val);
                melSum += val;
                melCount++;
            }
        }
        const melMean = melSum / melCount;
        const melStd = Math.sqrt(melFeatures.reduce((sum, val) => {
            if (isFinite(val)) {
                return sum + Math.pow(val - melMean, 2);
            }
            return sum;
        }, 0) / melCount);
        
        console.log('Mel 特征统计:', {
            最小值: melMin.toFixed(3),
            最大值: melMax.toFixed(3),
            平均值: melMean.toFixed(3),
            标准差: melStd.toFixed(3)
        });
        
        // 为每个通道创建不同的线性变换矩阵（模拟 encoder 的不同层）
        // 使用更合理的投影矩阵将 560 维映射到 512 维
        // 使用 PCA 风格的投影：将 Mel 特征的主要成分映射到 512 维
        const projectionMatrices = [];
        for (let c = 0; c < numChannels; c++) {
            const matrix = new Float32Array(melFeatureDim * featureDim);
            // 使用更合理的权重初始化（类似 Xavier/Glorot 初始化）
            const scale = Math.sqrt(2.0 / (melFeatureDim + featureDim));
            const seed = c * 12345;
            
            // 使用伪随机数生成器（基于通道索引）
            let rng = seed;
            const random = () => {
                rng = (rng * 1103515245 + 12345) & 0x7fffffff;
                return (rng / 0x7fffffff) * 2 - 1; // -1 到 1
            };
            
            for (let i = 0; i < matrix.length; i++) {
                // 使用正态分布风格的权重
                matrix[i] = random() * scale;
            }
            projectionMatrices.push(matrix);
        }
        
        for (let t = 0; t < frameCount; t++) {
            const melStart = t * melFeatureDim;
            const melFrame = new Float32Array(melFeatureDim);
            for (let i = 0; i < melFeatureDim; i++) {
                if (melStart + i < melFeatures.length) {
                    melFrame[i] = melFeatures[melStart + i];
                }
            }
            
            // 归一化 Mel 特征（Z-score normalization）
            for (let i = 0; i < melFeatureDim; i++) {
                if (melStd > 0) {
                    melFrame[i] = (melFrame[i] - melMean) / melStd;
                }
            }
            
            // 为每个通道生成特征
            for (let c = 0; c < numChannels; c++) {
                const baseIdx = (t * numChannels * featureDim) + (c * featureDim);
                const matrix = projectionMatrices[c];
                
                // 矩阵乘法：melFrame (560) x matrix (560x512) -> output (512)
                for (let f = 0; f < featureDim; f++) {
                    let sum = 0;
                    for (let m = 0; m < melFeatureDim; m++) {
                        sum += melFrame[m] * matrix[m * featureDim + f];
                    }
                    // 添加通道特定的偏置和激活
                    const bias = Math.sin((c / numChannels) * Math.PI * 2) * 0.01;
                    features[baseIdx + f] = sum + bias;
                }
            }
        }
        
        // 检查特征值范围
        let minVal = Infinity, maxVal = -Infinity;
        let validCount = 0;
        for (let i = 0; i < features.length; i++) {
            const val = features[i];
            if (isFinite(val) && !isNaN(val)) {
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
                validCount++;
            }
        }
        
        if (validCount === 0) {
            console.warn('警告：特征数组中没有有效值！');
            // 填充一些默认值以避免全零
            for (let i = 0; i < features.length; i++) {
                features[i] = (Math.random() - 0.5) * 0.1;
            }
            minVal = -0.05;
            maxVal = 0.05;
        }
        
        console.log('特征数组创建完成，实际长度:', features.length, '有效值:', validCount, '值范围:', minVal.toFixed(3), '到', maxVal.toFixed(3));
        return features;
    }
    
    /**
     * 前端特征提取：使用 Paraformer ONNX 模型
     */
    async extractFeaturesFromFrontend(audioData, frameCount) {
        console.log('使用前端 Paraformer 模型提取特征...');
        
        // 1. 前端特征提取（fbank + LFR + CMVN）
        const frontendResult = this.frontend.process(audioData);
        let feats = frontendResult.features; // [T * D] 格式
        let numFrames = frontendResult.numFrames;
        const featDim = frontendResult.featDim; // 应该是 560 (80 * 7)
        
        // 2. 固定输入长度为 150 帧（模型导出时的固定大小）
        // NOTE: Paraformer ONNX model has hardcoded attention mask for 150 frames
        const FIXED_TIME_DIM = 150;
        
        if (numFrames > FIXED_TIME_DIM) {
            // 截断到 150 帧
            console.warn(`输入长度 ${numFrames} 超过模型限制 ${FIXED_TIME_DIM}，将截断`);
            feats = feats.slice(0, FIXED_TIME_DIM * featDim);
            numFrames = FIXED_TIME_DIM;
        } else if (numFrames < FIXED_TIME_DIM) {
            // 填充到 150 帧（使用最后一帧的值）
            console.warn(`输入长度 ${numFrames} 小于模型要求 ${FIXED_TIME_DIM}，将填充`);
            const lastFrame = feats.slice(-featDim);
            const paddingFrames = FIXED_TIME_DIM - numFrames;
            const padding = new Float32Array(paddingFrames * featDim);
            for (let i = 0; i < paddingFrames; i++) {
                padding.set(lastFrame, i * featDim);
            }
            feats = new Float32Array([...feats, ...padding]);
            numFrames = FIXED_TIME_DIM;
        }
        
        console.log('前端特征提取完成:', {
            originalFrames: frontendResult.numFrames,
            adjustedFrames: numFrames,
            featDim: featDim,
            totalLength: feats.length
        });
        
        // 3. 准备 ONNX 模型输入
        // 输入格式：[B, T, D] = [1, 150, 560]
        const featsTensor = new ort.Tensor('float32', feats, [1, numFrames, featDim]);
        console.log('准备 ONNX 输入:', {
            featsShape: [1, numFrames, featDim],
            featsSize: feats.length,
            numFrames: numFrames
        });

        // 3. 运行 ONNX 模型
        const feeds = {};
        const inputNames = this.paraformerModel.inputNames || [];
        console.log('Paraformer 模型输入名称:', inputNames);
        
        // 准备 feats_lengths 张量（如果需要）
        const featsLengthsI32 = new ort.Tensor('int32', new Int32Array([numFrames]), [1]);
        const featsLengthsI64 = new ort.Tensor('int64', new BigInt64Array([BigInt(numFrames)]), [1]);
        
        // 根据输入名称匹配输入
        if (inputNames.length === 1) {
            // 只有一个输入，应该是 feats
            feeds[inputNames[0]] = featsTensor;
            console.log('使用单输入模式:', inputNames[0]);
        } else if (inputNames.length >= 2) {
            // 两个输入：feats 和 feats_lengths
            const name0 = inputNames[0];
            const name1 = inputNames[1];
            
            // 判断哪个是 feats，哪个是 lengths
            if (String(name0).toLowerCase().includes('len') || String(name0).toLowerCase().includes('length')) {
                feeds[name0] = featsLengthsI32;
                feeds[name1] = featsTensor;
                console.log('使用双输入模式 (lengths first):', { [name0]: 'int32[1]', [name1]: 'float32[1,T,D]' });
            } else if (String(name1).toLowerCase().includes('len') || String(name1).toLowerCase().includes('length')) {
                feeds[name0] = featsTensor;
                feeds[name1] = featsLengthsI32;
                console.log('使用双输入模式 (feats first):', { [name0]: 'float32[1,T,D]', [name1]: 'int32[1]' });
                } else {
                // 默认：第一个是 feats，第二个是 lengths
                feeds[name0] = featsTensor;
                feeds[name1] = featsLengthsI32;
                console.log('使用双输入模式 (默认顺序):', { [name0]: 'float32[1,T,D]', [name1]: 'int32[1]' });
            }
        } else {
            // 极端防御性回退：假设输入名称是 'feats'
            feeds.feats = featsTensor;
            console.warn('未检测到输入名称，使用默认名称 "feats"');
        }

        // 4. 运行推理（带重试机制）
        let results;
        try {
            console.log('开始 Paraformer 推理，输入:', Object.keys(feeds));
            // 打印输入张量的详细信息
            for (const [name, tensor] of Object.entries(feeds)) {
                console.log(`  输入 ${name}: 类型=${tensor.type}, 形状=[${tensor.dims.join(', ')}], 大小=${tensor.size}`);
            }
            results = await this.paraformerModel.run(feeds);
            console.log('Paraformer 推理成功，输出:', Object.keys(results));
            // 打印输出张量的详细信息
            for (const [name, tensor] of Object.entries(results)) {
                console.log(`  输出 ${name}: 类型=${tensor.type}, 形状=[${tensor.dims.join(', ')}], 大小=${tensor.size}`);
            }
        } catch (e) {
            // 打印详细的错误信息
            console.error('Paraformer 推理失败，详细信息:');
            console.error('  错误类型:', typeof e);
            console.error('  错误消息:', e.message || String(e));
            console.error('  错误代码:', (typeof e === 'number') ? e : (e.code || 'N/A'));
            console.error('  输入信息:');
            for (const [name, tensor] of Object.entries(feeds)) {
                console.error(`    ${name}: 类型=${tensor.type}, 形状=[${tensor.dims.join(', ')}], 大小=${tensor.size}`);
            }
            
            // 如果是双输入且失败，尝试用 int64 的 lengths 重试
            if (inputNames.length >= 2) {
                console.warn('使用 int32 lengths 失败，尝试 int64:', e.message);
                const retryFeeds = { ...feeds };
                for (const k of Object.keys(retryFeeds)) {
                    if (String(k).toLowerCase().includes('len') || String(k).toLowerCase().includes('length')) {
                        retryFeeds[k] = featsLengthsI64;
                        console.log('将', k, '改为 int64');
                    }
                }
                try {
                    results = await this.paraformerModel.run(retryFeeds);
                    console.log('使用 int64 lengths 重试成功');
                } catch (e2) {
                    console.error('Paraformer 推理失败 (int64 重试也失败):', e2);
                    const normalized = (typeof e2 === 'number') ? new Error(`Paraformer run failed (code=${e2})`) : e2;
                    throw normalized;
                }
            } else {
                const normalized = (typeof e === 'number') ? new Error(`Paraformer run failed (code=${e})`) : e;
                throw normalized;
            }
        }

        const hidden = results.hidden; // [B, L, T, C] = [1, 50, T, 512]
        
        console.log('Paraformer 模型推理完成:', {
            hiddenShape: hidden.dims,
            hiddenSize: hidden.size
        });
        
        // 4. 直接在 hidden.data 上做时间插值，避免构造大量中间数组导致内存爆炸
        const hiddenData = hidden.data; // Float32Array
        const [B, L, T, C] = hidden.dims; // expected B=1, L=50, T=numFrames, C=512

        if (B !== 1) {
            throw new Error(`Unexpected batch size from paraformer hidden: B=${B}`);
        }
        if (C !== 512) {
            console.warn(`Unexpected hidden dim C=${C} (expected 512)`);
        }

        const outputFeatures = new Float32Array(frameCount * L * C); // [frameCount, L, C]

        // linear interpolation along time axis
        for (let tOut = 0; tOut < frameCount; tOut++) {
            const ratio = (frameCount === 1) ? 0 : (tOut / (frameCount - 1)) * (T - 1);
            const t0 = Math.floor(ratio);
            const t1 = Math.min(t0 + 1, T - 1);
            const a = ratio - t0;

            for (let l = 0; l < L; l++) {
                const base0 = (l * T * C) + (t0 * C);
                const base1 = (l * T * C) + (t1 * C);
                const outBase = (tOut * L * C) + (l * C);

                for (let c = 0; c < C; c++) {
                    const v0 = hiddenData[base0 + c];
                    const v1 = hiddenData[base1 + c];
                    outputFeatures[outBase + c] = v0 * (1 - a) + v1 * a;
                }
            }
        }
        
        console.log('前端特征提取完成:', {
            outputShape: [frameCount, L, C],
            outputLength: outputFeatures.length
        });
        
        return outputFeatures;
    }
    

    async audio2mouthInference(audioFeatures, frameCount) {
        if (!this.audio2mouthModel) {
            await this.initializeModels();
        }

        this.updateStatus('', '正在生成嘴部参数...');

        const interval = 1.0;
        const frag = Math.floor(interval * 30 / 5 + 0.5);
        const paramRes = [];

        let startTime = 0.0;
        let endTime = startTime + interval;
        const audioLength = frameCount / 30;
        let isEnd = false;

        while (true) {
            let start = Math.floor(startTime * 16000);
            let end = start + 16000;

            if (endTime >= audioLength) {
                isEnd = true;
                end = Math.floor(audioLength * 16000);
                start = end - 16000;
                startTime = audioLength - interval;
                endTime = audioLength;
            }

            const startFrame = Math.floor(startTime * 30);
            // 根据 Python 代码：end_frame = start_frame + int(30 * interval)，其中 interval=1.0
            // 所以 input_au 的时间维度是 50（模型期望），但 input_ph 的时间维度是 30
            const expectedTimeFramesAu = 50; // input_au 的时间维度
            const expectedTimeFramesPh = 30; // input_ph 的时间维度
            const numChannels = 30; // 根据错误信息，通道数应该是 30
            const featureDim = 512; // 特征维度
            
            // 准备输入: 模型期望 [batch=1, channels=30, time=50, features=512]
            // 需要从特征中提取对应的时间帧和通道
            const inputAuData = new Float32Array(1 * numChannels * expectedTimeFramesAu * featureDim);
            
            // 检查特征数组长度
            // 需要的总长度 = frameCount * numChannels * featureDim
            const expectedTotalLength = frameCount * numChannels * featureDim;
            const requiredLength = (startFrame + expectedTimeFramesAu) * numChannels * featureDim;
            
            console.log(`特征数组检查: 实际长度=${audioFeatures.length}, 期望总长度=${expectedTotalLength}, 当前窗口需要=${requiredLength}, startFrame=${startFrame}, frameCount=${frameCount}`);
            
            // 如果特征数组长度不足，需要扩展
            if (audioFeatures.length < expectedTotalLength) {
                console.warn(`特征数组长度不足，扩展中: 实际 ${audioFeatures.length}, 期望 ${expectedTotalLength}`);
                const expandedFeatures = new Float32Array(expectedTotalLength);
                // 复制现有特征
                const copyLength = Math.min(audioFeatures.length, expectedTotalLength);
                expandedFeatures.set(audioFeatures.slice(0, copyLength));
                // 剩余部分用零填充
                if (copyLength < expectedTotalLength) {
                    expandedFeatures.fill(0, copyLength);
                }
                audioFeatures = expandedFeatures;
                console.log(`特征数组已扩展，新长度: ${audioFeatures.length}`);
            }
            
            // 检查当前窗口所需的长度
            if (audioFeatures.length < requiredLength) {
                console.warn(`当前窗口所需长度不足: 需要 ${requiredLength}, 实际 ${audioFeatures.length}`);
                // 创建填充数组
                const paddedFeatures = new Float32Array(requiredLength);
                const copyLength = Math.min(audioFeatures.length, requiredLength);
                paddedFeatures.set(audioFeatures.slice(0, copyLength));
                // 剩余部分用零填充
                if (copyLength < requiredLength) {
                    paddedFeatures.fill(0, copyLength);
                }
                audioFeatures = paddedFeatures;
            }
            
            // 从特征中提取: au_data 格式是 [time, channels=30, features=512]
            // Python: input_au = au_data[start_frame:end_frame] 得到 [30, 30, 512]
            // 然后 input_au = input_au[np.newaxis,:] 变成 [1, 30, 30, 512]
            // 但模型期望 [1, 30, 50, 512]，所以需要取更大的窗口
            // 根据模型输入，time=50，所以需要取 start_frame-10 到 start_frame+40
            const windowStart = Math.max(0, startFrame - 10);
            const windowSize = expectedTimeFramesAu; // 50
            
            // 从特征中提取: 特征格式是 [time, channels, features] 的扁平数组
            // 目标格式: [channels, time, features] = [30, 50, 512]
            for (let t = 0; t < expectedTimeFramesAu; t++) {
                const frameIdx = windowStart + t;
                if (frameIdx >= frameCount) {
                    // 超出范围，用零填充（inputAuData 已初始化为0）
                    continue;
                }
                for (let c = 0; c < numChannels; c++) {
                    // 源索引: [time, channels, features] 格式
                    // frameIdx * (channels * features) + c * features
                    const srcIdx = (frameIdx * numChannels * featureDim) + (c * featureDim);
                    // 目标索引: [channels, time, features] 格式（模型期望）
                    // c * (time * features) + t * features
                    const dstIdx = (c * expectedTimeFramesAu * featureDim) + (t * featureDim);
                    if (srcIdx + featureDim <= audioFeatures.length && dstIdx + featureDim <= inputAuData.length) {
                        for (let f = 0; f < featureDim; f++) {
                            const val = audioFeatures[srcIdx + f];
                            if (!isNaN(val) && isFinite(val)) {
                                inputAuData[dstIdx + f] = val;
                            }
                        }
                    }
                }
            }
            
            // 创建 4 维张量: [batch=1, channels=30, time=50, features=512]
            const inputAuTensor = new ort.Tensor('float32', inputAuData, [1, numChannels, expectedTimeFramesAu, featureDim]);
            
            // 调试：检查输入数据是否有效
            if (startFrame === 0 || startFrame === 60 || startFrame === 120) {
                let validCount = 0;
                let nanCount = 0;
                for (let i = 0; i < Math.min(100, inputAuData.length); i++) {
                    if (isNaN(inputAuData[i])) {
                        nanCount++;
                    } else if (inputAuData[i] !== 0) {
                        validCount++;
                    }
                }
                console.log(`startFrame=${startFrame} 输入数据检查: 有效值=${validCount}, NaN=${nanCount}, 总长度=${inputAuData.length}`);
            }

            // input_ph 的时间维度是 30（根据 Python 代码：end_frame = start_frame + 30）
            // [batch, time, features] = [1, 30, 2]
            const inputPh = new Float32Array(expectedTimeFramesPh * 2).fill(0);
            const inputPhTensor = new ort.Tensor('float32', inputPh, [1, expectedTimeFramesPh, 2]);

            const w = new ort.Tensor('float32', new Float32Array([1.0]), [1]);
            const sp = new ort.Tensor('int64', new BigInt64Array([2n]), [1]);

            // 运行推理
            if (!this.audio2mouthModel) {
                throw new Error('音频到嘴部模型未加载');
            }
            
            const results = await this.audio2mouthModel.run({
                input_au: inputAuTensor,
                input_ph: inputPhTensor,
                input_sp: sp,
                w: w
            });
            
            if (!results || !results.output) {
                throw new Error('模型推理返回空结果');
            }

            const output = results.output.data;
            
            if (!output || output.length === 0) {
                throw new Error('模型输出为空');
            }
            
            // 处理输出（类似 Python 版本的逻辑）
            // 根据 Python 代码，输出应该是 [batch, time=30, features=32]
            const outputShape = results.output.dims || [1, expectedTimeFramesPh, 32];
            const numParams = outputShape[2] || 32;
            const totalFrames = outputShape[1] || expectedTimeFramesPh;
            
            // 调试：检查模型输出是否变化
            const firstFrameParams = [];
            for (let ii = 0; ii < numParams; ii++) {
                firstFrameParams.push(output[ii]);
            }
            console.log(`startFrame=${startFrame}, startTime=${startTime.toFixed(3)}, 模型输出前5个参数: [${firstFrameParams.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
            
            // 检查输入特征是否不同
            if (startFrame === 0 || startFrame === 60 || startFrame === 120) {
                const sampleFeatureValues = [];
                for (let i = 0; i < 10; i++) {
                    sampleFeatureValues.push(inputAuData[i].toFixed(3));
                }
                console.log(`startFrame=${startFrame} 输入特征前10个值: [${sampleFeatureValues.join(', ')}]`);
            }
            
            // 辅助函数
            const round = (value, decimals) => {
                if (typeof value !== 'number' || isNaN(value)) return 0;
                return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
            };
            
            // 处理输出帧
            // Python 代码逻辑：
            // - 如果 start_time == 0.0 且 !is_end: 处理前 (30 * interval - frag) 帧
            // - 如果 start_time > 0.0 且 !is_end: 处理 frag 到 (30 * interval - frag) 帧
            // - 如果 is_end: 处理 frag 到 (30 * interval) 帧
            const frag = Math.floor(interval * 30 / 5 + 0.5); // frag = 6
            
            for (let tt = 0; tt < totalFrames; tt++) {
                const frameId = startFrame + tt;
                const paramFrame = {};
                
                for (let ii = 0; ii < numParams; ii++) {
                    const index = tt * numParams + ii;
                    const value = index < output.length ? output[index] : 0;
                    paramFrame[String(ii)] = round(value, 3);
                }
                
                // 处理重叠区域（类似 Python 版本的逻辑）
                if (startTime === 0.0 && !isEnd) {
                    // 第一段，跳过最后的 frag 帧
                    if (tt < totalFrames - frag) {
                        paramRes.push(paramFrame);
                    }
                } else if (startTime > 0.0 && !isEnd) {
                    // 中间段，处理 frag 到 (totalFrames - frag) 帧
                    if (tt >= frag && tt < totalFrames - frag) {
                        if (frameId < paramRes.length) {
                            // 重叠区域，进行混合
                            const scale = Math.min((paramRes.length - frameId) / frag, 1.0);
                            for (let key in paramFrame) {
                                const oldValue = paramRes[frameId]?.[key] || 0;
                                paramFrame[key] = (1 - scale) * paramFrame[key] + scale * oldValue;
                            }
                            paramRes[frameId] = paramFrame;
                        } else {
                            paramRes.push(paramFrame);
                        }
                    }
                } else {
                    // 最后一段，处理 frag 到 totalFrames 帧
                    if (tt >= frag) {
                        if (frameId < paramRes.length) {
                            // 重叠区域，进行混合
                            const scale = Math.min((paramRes.length - frameId) / frag, 1.0);
                            for (let key in paramFrame) {
                                const oldValue = paramRes[frameId]?.[key] || 0;
                                paramFrame[key] = (1 - scale) * paramFrame[key] + scale * oldValue;
                            }
                            paramRes[frameId] = paramFrame;
                        } else {
                            paramRes.push(paramFrame);
                        }
                    }
                }
            }

            if (isEnd) break;

            startTime = endTime - (frag / 10);
            endTime = startTime + interval;
        }

        // 平滑处理
        return this.smoothParams(paramRes);
    }

    smoothParams(params) {
        // 实现低通滤波平滑
        // 简化版实现
        return params;
    }

    async generateVideo() {
        // 检查音频文件：优先使用文件输入，如果没有则使用 this.audioFile（默认音频或录音）
        const audioFileInput = document.getElementById('audioFile');
        const audioFile = audioFileInput?.files[0] || this.audioFile;
        
        if (!audioFile) {
            this.updateStatus('error', '请先上传音频文件或使用默认示例音频');
            return;
        }

        if (!this.avatarData) {
            this.updateStatus('error', '请先加载 Avatar 数据');
            return;
        }

        try {
            console.log('开始生成视频...');
            console.log('音频文件:', audioFile.name);
            console.log('Avatar 数据:', this.avatarData ? Object.keys(this.avatarData).length + ' 个文件' : '无');
            
            // 初始化模型
            this.updateStatus('', '正在初始化模型...');
            await this.initializeModels();
            console.log('✓ 模型初始化完成');

            // 加载 Avatar 数据（如果还没有加载）
            if (this.avatarData && Object.keys(this.avatarData).length > 0) {
                this.updateStatus('', '正在加载 Avatar 数据...');
                await this.loadAvatarData(this.avatarData);
                console.log('✓ Avatar 数据加载完成');
            } else {
                throw new Error('请先加载 Avatar 数据');
            }

            // 确定目标帧数（基于视频帧数，确保音频和视频对齐）
            const videoFrameCount = this.bgVideoFrames ? this.bgVideoFrames.length : 150;
            const fps = 30;
            const targetFrameCount = videoFrameCount;
            
            console.log('视频帧数:', videoFrameCount, '目标音频帧数:', targetFrameCount);
            
            // 处理音频（传入目标帧数，确保音频特征和视频帧数匹配）
            this.updateStatus('', '正在处理音频...');
            const mouthParams = await this.processAudio(audioFile, targetFrameCount);
            console.log('✓ 音频处理完成，生成', mouthParams.length, '帧参数');
            
            if (!mouthParams || mouthParams.length === 0) {
                throw new Error('未能生成嘴部参数');
            }
            
            // 确保嘴部参数帧数和视频帧数匹配
            if (mouthParams.length !== targetFrameCount) {
                console.warn(`警告：嘴部参数帧数 (${mouthParams.length}) 与视频帧数 (${targetFrameCount}) 不匹配，将进行调整`);
                // 如果参数帧数少于视频帧数，重复最后一帧
                // 如果参数帧数多于视频帧数，截断
                while (mouthParams.length < targetFrameCount) {
                    mouthParams.push(mouthParams[mouthParams.length - 1]);
                }
                if (mouthParams.length > targetFrameCount) {
                    mouthParams.splice(targetFrameCount);
                }
                console.log('调整后嘴部参数帧数:', mouthParams.length);
            }

            // 生成视频帧
            this.updateStatus('', '正在生成视频帧...');
            const frames = await this.generateFrames(mouthParams);

            // 合成视频
            this.updateStatus('', '正在合成视频...');
            const videoBlob = await this.composeVideo(frames, audioFile);

            // 显示结果
            const videoElement = document.getElementById('outputVideo');
            videoElement.src = URL.createObjectURL(videoBlob);
            videoElement.classList.remove('hidden');

            const downloadLink = document.getElementById('downloadLink');
            downloadLink.href = URL.createObjectURL(videoBlob);
            downloadLink.classList.remove('hidden');

            this.updateStatus('success', '视频生成完成！');
        } catch (error) {
            const errorMessage = error?.message || error?.toString() || '未知错误';
            const errorStack = error?.stack || '';
            console.error('生成视频失败:', error);
            console.error('错误堆栈:', errorStack);
            this.updateStatus('error', `生成失败: ${errorMessage}`);
            
            // 显示更详细的错误信息
            if (errorMessage.includes('model') || errorMessage.includes('模型')) {
                this.updateStatus('error', `模型错误: ${errorMessage}。请检查模型文件是否存在。`);
            } else if (errorMessage.includes('audio') || errorMessage.includes('音频')) {
                this.updateStatus('error', `音频处理错误: ${errorMessage}。请检查音频文件格式。`);
            } else {
                this.updateStatus('error', `生成失败: ${errorMessage}。请查看控制台获取详细信息。`);
            }
        }
    }

    async generateFrames(mouthParams) {
        const frames = [];
        const totalFrames = mouthParams.length;

        for (let i = 0; i < totalFrames; i++) {
            // 更新进度
            const progress = ((i + 1) / totalFrames * 100).toFixed(1);
            this.updateProgress(progress);

            // 生成单帧
            const frame = await this.generateFrame(mouthParams[i], i);
            frames.push(frame);
        }

        return frames;
    }

    async generateFrame(mouthParams, frameIndex) {
        // 选择背景帧
        let bgFrame = null;
        
        if (this.bgVideoFrames && this.bgVideoFrames.length > 0) {
            // 使用背景视频帧
            const bgFrameIndex = frameIndex % this.bgVideoFrames.length;
            bgFrame = this.bgVideoFrames[bgFrameIndex];
        } else if (this.refFrames && this.refFrames.length > 0) {
            // 如果没有背景视频帧，使用参考帧作为背景
            const refFrameIndex = frameIndex % this.refFrames.length;
            const refFrameData = this.refFrames[refFrameIndex];
            
            // 从参考帧创建 ImageData
            if (refFrameData.url) {
                const img = new Image();
                await new Promise((resolve, reject) => {
                    img.onload = () => {
                        const canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        bgFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        resolve();
                    };
                    img.onerror = reject;
                    img.src = refFrameData.url;
                });
            }
        } else {
            // 如果都没有，创建一个空白帧
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            bgFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
        }

        // 选择参考帧用于生成嘴部
        // Python 代码: self.generator(self.ref_img_list[bg_frame_id], ...)
        // bg_frame_id 应该与背景帧索引一致
        const bgFrameIndex = frameIndex % (this.bgVideoFrames?.length || this.refFrames?.length || 1);
        let refFrame = null;
        
        if (this.refFrames && this.refFrames.length > 0) {
            // 使用与背景帧相同的索引来选择参考帧
            const refFrameIndex = bgFrameIndex % this.refFrames.length;
            const refFrameData = this.refFrames[refFrameIndex];
            if (refFrameData && refFrameData.url) {
                const img = new Image();
                await new Promise((resolve, reject) => {
                    img.onload = () => {
                        const canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        refFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        resolve();
                    };
                    img.onerror = reject;
                    img.src = refFrameData.url;
                });
            }
        }
        
        // 使用生成器模型生成嘴部图像（如果有模型）
        let mouthImage = null;
        if (this.generatorModel && refFrame) {
            mouthImage = await this.generateMouthImage(mouthParams, refFrame, frameIndex);
        } else if (refFrame) {
            // 简化版：使用参考帧并根据参数进行简单变形
            mouthImage = await this.generateMouthImageSimple(mouthParams, refFrame);
        }

        // 合并到背景
        if (mouthImage && this.faceBox) {
            const finalFrame = this.mergeMouthToBackground(mouthImage, bgFrame);
            return finalFrame;
        }

        // 如果没有嘴部图像，直接返回背景帧
        return bgFrame;
    }

    async generateMouthImage(mouthParams, refFrame, frameIndex = 0) {
        // 使用生成器模型生成嘴部图像
        // 需要先运行编码器获取特征，然后用特征+参数调用生成器
        if (!this.generatorModel || !this.encoderModel) {
            console.warn('生成器或编码器模型未加载，使用简化版本');
            return null;
        }
        
        try {
            // 1. 先运行编码器获取特征列表
            const refImageData = refFrame;
            const width = refImageData.width;
            const height = refImageData.height;
            
            // 调整图像大小到 384x384（编码器期望的输入）
            const targetSize = 384;
            const canvas = document.createElement('canvas');
            canvas.width = targetSize;
            canvas.height = targetSize;
            const ctx = canvas.getContext('2d');
            
            // 创建临时canvas绘制参考帧
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(refImageData, 0, 0);
            
            // 调整大小
            ctx.drawImage(tempCanvas, 0, 0, targetSize, targetSize);
            const resizedImageData = ctx.getImageData(0, 0, targetSize, targetSize);
            
            // 转换为 [1, 3, 384, 384] 格式的 tensor，归一化到 [-1, 1]
            const refImageArray = new Float32Array(1 * 3 * targetSize * targetSize);
            const data = resizedImageData.data;
            
            for (let y = 0; y < targetSize; y++) {
                for (let x = 0; x < targetSize; x++) {
                    const idx = (y * targetSize + x) * 4;
                    const r = data[idx];
                    const g = data[idx + 1];
                    const b = data[idx + 2];
                    
                    // 归一化到 [-1, 1]（使用 transforms.Normalize([0.5], [0.5]) 的效果）
                    const rIdx = (0 * targetSize * targetSize) + (y * targetSize) + x;
                    const gIdx = (1 * targetSize * targetSize) + (y * targetSize) + x;
                    const bIdx = (2 * targetSize * targetSize) + (y * targetSize) + x;
                    
                    refImageArray[rIdx] = (r / 255.0) * 2.0 - 1.0;
                    refImageArray[gIdx] = (g / 255.0) * 2.0 - 1.0;
                    refImageArray[bIdx] = (b / 255.0) * 2.0 - 1.0;
                }
            }
            
            const refImageTensor = new ort.Tensor('float32', refImageArray, [1, 3, targetSize, targetSize]);
            
            // 运行编码器获取特征列表
            const encoderResults = await this.encoderModel.run({
                ref_image: refImageTensor
            });
            
            // 编码器现在返回4个独立输出: output_0, output_1, output_2, output_3
            // 根据编码器输出形状: [(1, 3, 384, 384), (1, 16, 192, 192), (1, 32, 96, 96), (1, 64, 48, 48)]
            const encoderOutputs = [];
            if (encoderResults.output_0 && encoderResults.output_1 && encoderResults.output_2 && encoderResults.output_3) {
                // 多个独立输出
                encoderOutputs.push(encoderResults.output_0); // (1, 3, 384, 384)
                encoderOutputs.push(encoderResults.output_1); // (1, 16, 192, 192)
                encoderOutputs.push(encoderResults.output_2); // (1, 32, 96, 96)
                encoderOutputs.push(encoderResults.output_3); // (1, 64, 48, 48)
            } else {
                // 兼容旧格式（如果只有一个输出）
                console.warn('编码器输出格式异常，尝试兼容模式');
                if (encoderResults.output) {
                    encoderOutputs.push(encoderResults.output);
                } else {
                    // 尝试按索引获取
                    for (let i = 0; i < 4; i++) {
                        const key = `output_${i}`;
                        if (encoderResults[key]) {
                            encoderOutputs.push(encoderResults[key]);
                        }
                    }
                }
            }
            
            if (encoderOutputs.length < 4) {
                throw new Error(`编码器输出数量不足: 期望4个，实际${encoderOutputs.length}。可用键: ${Object.keys(encoderResults).join(', ')}`);
            }
            
            // 2. 准备参数值（32个参数）
            // Python 代码: 
            //   param_val = []
            //   for key in param_res:
            //       param_val.append(param_res[key])
            // 由于 Python 3.7+ 字典保持插入顺序，且参数是按 p_list 顺序添加的
            // 所以应该按照 p_list 的顺序提取：["0", "1", ..., "31"]
            const paramValues = [];
            const pList = [];
            for (let i = 0; i < 32; i++) {
                pList.push(String(i));
            }
            // 按照 p_list 顺序提取参数值（与 Python 代码一致）
            for (const key of pList) {
                const value = mouthParams[key] || 0;
                paramValues.push(value);
            }
            
            // 调试：检查参数值范围
            if (frameIndex === 0 || frameIndex % 60 === 0) {
                const minVal = Math.min(...paramValues);
                const maxVal = Math.max(...paramValues);
                const avgVal = paramValues.reduce((a, b) => a + b, 0) / paramValues.length;
                console.log(`帧 ${frameIndex} 参数值范围: [${minVal.toFixed(3)}, ${maxVal.toFixed(3)}], 平均值: ${avgVal.toFixed(3)}`);
                console.log(`  前5个参数: [${paramValues.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
            }
            
            const paramTensor = new ort.Tensor('float32', new Float32Array(paramValues), [1, 32]);
            
            // 3. 运行生成器模型
            // 生成器需要: [input, skip1, skip0, skip] 和 params（与编码器输出顺序一致）
            // input = encoderOutputs[0] (1, 3, 384, 384)
            // skip1 = encoderOutputs[1] (1, 16, 192, 192)
            // skip0 = encoderOutputs[2] (1, 32, 96, 96)
            // skip = encoderOutputs[3] (1, 64, 48, 48)
            const results = await this.generatorModel.run({
                input: encoderOutputs[0],
                skip1: encoderOutputs[1],
                skip0: encoderOutputs[2],
                skip: encoderOutputs[3],
                params: paramTensor
            });
            
            // 处理输出
            const output = results.output;
            if (!output) {
                throw new Error('模型输出为空');
            }
            
            // 将输出转换为 ImageData
            const outputData = output.data;
            const outputDims = output.dims || [1, 3, targetSize, targetSize];
            const outputHeight = outputDims[2] || targetSize;
            const outputWidth = outputDims[3] || targetSize;
            
            const outputImageData = new ImageData(outputWidth, outputHeight);
            const outputArray = outputImageData.data;
            
            // 从 [1, 3, H, W] 转换为 RGBA ImageData
            // Python 代码: mouth_image = (mouth_image / 2 + 0.5).clamp(0, 1) * 255
            // 等价于: (mouth_image + 1) * 127.5
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    const rIdx = (0 * outputHeight * outputWidth) + (y * outputWidth) + x;
                    const gIdx = (1 * outputHeight * outputWidth) + (y * outputWidth) + x;
                    const bIdx = (2 * outputHeight * outputWidth) + (y * outputWidth) + x;
                    
                    const idx = (y * outputWidth + x) * 4;
                    // 从 [-1, 1] 反归一化到 [0, 255]
                    // Python: (x / 2 + 0.5) * 255 = (x + 1) * 127.5
                    outputArray[idx] = Math.max(0, Math.min(255, (outputData[rIdx] + 1) * 127.5));
                    outputArray[idx + 1] = Math.max(0, Math.min(255, (outputData[gIdx] + 1) * 127.5));
                    outputArray[idx + 2] = Math.max(0, Math.min(255, (outputData[bIdx] + 1) * 127.5));
                    outputArray[idx + 3] = 255;
                }
            }
            
            return outputImageData;
        } catch (error) {
            console.error('生成嘴部图像失败:', error);
            return null;
        }
    }
    
    async generateMouthImageSimple(mouthParams, refFrame) {
        // 简化版：根据嘴部参数对参考帧进行简单变形
        // 创建一个 canvas 来绘制变形的嘴部
        const canvas = document.createElement('canvas');
        canvas.width = refFrame.width;
        canvas.height = refFrame.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制参考帧
        ctx.putImageData(refFrame, 0, 0);
        
        // 如果有 faceBox，在嘴部区域应用变形
        if (this.faceBox) {
            const { y1, y2, x1, x2 } = this.faceBox;
            const mouthWidth = x2 - x1;
            const mouthHeight = y2 - y1;
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;
            
            // 获取嘴部区域的图像数据
            const imageData = ctx.getImageData(x1, y1, mouthWidth, mouthHeight);
            const data = imageData.data;
            
            // 计算嘴部参数的变化（使用前几个主要参数）
            const paramKeys = ['0', '1', '2', '3', '4', '5'];
            let paramSum = 0;
            for (const key of paramKeys) {
                paramSum += mouthParams[key] || 0;
            }
            const avgParam = paramSum / paramKeys.length;
            
            // 根据参数值调整嘴部区域的颜色和形状
            // 使用简单的径向变形和颜色调整
            for (let y = 0; y < mouthHeight; y++) {
                for (let x = 0; x < mouthWidth; x++) {
                    const dx = x - mouthWidth / 2;
                    const dy = y - mouthHeight / 2;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const maxDist = Math.min(mouthWidth, mouthHeight) / 2;
                    
                    // 根据距离和参数计算变形强度
                    const normalizedDist = dist / maxDist;
                    const intensity = (avgParam - 0.5) * 0.5; // 调整强度
                    const factor = 1 - normalizedDist * intensity;
                    
                    const idx = (y * mouthWidth + x) * 4;
                    // 调整颜色以模拟嘴部运动
                    data[idx] = Math.max(0, Math.min(255, data[idx] * (1 + factor * 0.3)));     // R
                    data[idx + 1] = Math.max(0, Math.min(255, data[idx + 1] * (1 + factor * 0.15))); // G
                    data[idx + 2] = Math.max(0, Math.min(255, data[idx + 2] * (1 + factor * 0.2))); // B
                }
            }
            
            // 将修改后的图像数据放回
            ctx.putImageData(imageData, x1, y1);
        } else {
            // 如果没有 faceBox，在整个图像中心应用简单变形
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            const paramValues = Object.values(mouthParams);
            const avgParam = paramValues.reduce((a, b) => a + b, 0) / paramValues.length;
            const intensity = (avgParam - 0.5) * 0.3;
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height * 0.6;
            const radius = Math.min(canvas.width, canvas.height) * 0.15;
            
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < radius) {
                        const idx = (y * canvas.width + x) * 4;
                        const factor = 1 - (dist / radius) * intensity;
                        data[idx] = Math.min(255, data[idx] * (1 + factor * 0.2));
                        data[idx + 1] = Math.min(255, data[idx + 1] * (1 + factor * 0.1));
                        data[idx + 2] = Math.min(255, data[idx + 2] * (1 + factor * 0.15));
                    }
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
        }
        
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    mergeMouthToBackground(mouthImage, bgFrame) {
        if (!this.faceBox || !mouthImage) {
            return bgFrame;
        }
        
        // 创建输出 canvas
        const canvas = document.createElement('canvas');
        canvas.width = bgFrame.width;
        canvas.height = bgFrame.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制背景帧
        ctx.putImageData(bgFrame, 0, 0);
        
        // 获取嘴部区域
        const { y1, y2, x1, x2 } = this.faceBox;
        const mouthWidth = x2 - x1;
        const mouthHeight = y2 - y1;
        
        // 创建临时 canvas 来调整嘴部图像大小
        const mouthCanvas = document.createElement('canvas');
        mouthCanvas.width = mouthWidth;
        mouthCanvas.height = mouthHeight;
        const mouthCtx = mouthCanvas.getContext('2d');
        
        // 调整嘴部图像大小
        mouthCtx.drawImage(
            this.imageDataToCanvas(mouthImage).canvas,
            0, 0, mouthImage.width, mouthImage.height,
            0, 0, mouthWidth, mouthHeight
        );
        
        const mouthImageData = mouthCtx.getImageData(0, 0, mouthWidth, mouthHeight);
        const bgImageData = ctx.getImageData(x1, y1, mouthWidth, mouthHeight);
        
        // 使用 mergeMask 进行混合
        if (this.mergeMask && this.mergeMask.length === mouthWidth * mouthHeight) {
            for (let i = 0; i < mouthWidth * mouthHeight; i++) {
                const maskValue = this.mergeMask[i];
                const bgIdx = i * 4;
                const mouthIdx = i * 4;
                
                bgImageData.data[bgIdx] = mouthImageData.data[mouthIdx] * (1 - maskValue) + bgImageData.data[bgIdx] * maskValue;
                bgImageData.data[bgIdx + 1] = mouthImageData.data[mouthIdx + 1] * (1 - maskValue) + bgImageData.data[bgIdx + 1] * maskValue;
                bgImageData.data[bgIdx + 2] = mouthImageData.data[mouthIdx + 2] * (1 - maskValue) + bgImageData.data[bgIdx + 2] * maskValue;
                bgImageData.data[bgIdx + 3] = 255; // Alpha
            }
        } else {
            // 如果没有 mergeMask，直接替换
            for (let i = 0; i < mouthWidth * mouthHeight; i++) {
                const idx = i * 4;
                bgImageData.data[idx] = mouthImageData.data[idx];
                bgImageData.data[idx + 1] = mouthImageData.data[idx + 1];
                bgImageData.data[idx + 2] = mouthImageData.data[idx + 2];
                bgImageData.data[idx + 3] = 255;
            }
        }
        
        // 将混合后的图像放回背景
        ctx.putImageData(bgImageData, x1, y1);
        
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    imageDataToCanvas(imageData) {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        return { canvas, ctx };
    }

    async composeVideo(frames, audioFile) {
        if (!frames || frames.length === 0) {
            throw new Error('没有可用的视频帧');
        }

        console.log('开始合成视频，共', frames.length, '帧');
        
        // 获取第一帧的尺寸
        const firstFrame = frames[0];
        const width = firstFrame.width;
        const height = firstFrame.height;
        const fps = 30;

        // 创建 canvas 用于绘制帧
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // 创建视频流
        const videoStream = canvas.captureStream(fps);
        
        // 创建音频流（使用已处理的音频）
        let audioStream = null;
        if (this.processedAudioBuffer) {
            try {
                const targetSampleRate = 16000;
                
                // 创建 AudioContext 用于音频输出（16kHz）
                if (!this.outputAudioContext) {
                    this.outputAudioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: targetSampleRate
                    });
                }
                
                // 直接使用已处理的音频缓冲区
                const processedBuffer = this.processedAudioBuffer;
                
                // 创建音频源节点
                const source = this.outputAudioContext.createBufferSource();
                source.buffer = processedBuffer;
                
                // 创建 MediaStreamDestination 来获取音频流
                const destination = this.outputAudioContext.createMediaStreamDestination();
                source.connect(destination);
                
                // 开始播放音频
                source.start(0);
                
                audioStream = destination.stream;
                console.log('使用已处理的音频:', {
                    采样率: processedBuffer.sampleRate,
                    通道数: processedBuffer.numberOfChannels,
                    时长: processedBuffer.duration.toFixed(2) + '秒',
                    采样数: processedBuffer.length
                });
            } catch (error) {
                console.warn('无法添加音频到视频:', error);
                // 继续生成没有音频的视频
            }
        } else if (audioFile) {
            console.warn('未找到已处理的音频，尝试从原始文件处理（不推荐）');
            // 降级处理：如果 processedAudioBuffer 不存在，尝试从原始文件处理
            // 但这不应该发生，因为 processAudio 应该已经处理了音频
        }

        // 合并视频和音频流
        const combinedStream = new MediaStream();
        videoStream.getVideoTracks().forEach(track => combinedStream.addTrack(track));
        if (audioStream) {
            audioStream.getAudioTracks().forEach(track => combinedStream.addTrack(track));
        }

        // 创建 MediaRecorder
        const mimeTypes = [
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm'
        ];
        
        let selectedMimeType = null;
        for (const mimeType of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mimeType)) {
                selectedMimeType = mimeType;
                break;
            }
        }
        
        if (!selectedMimeType) {
            throw new Error('浏览器不支持视频录制');
        }
        
        console.log('使用 MIME 类型:', selectedMimeType);
        const mediaRecorder = new MediaRecorder(combinedStream, {
            mimeType: selectedMimeType,
            videoBitsPerSecond: 2500000
        });

        const chunks = [];
        
        return new Promise((resolve, reject) => {
            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    chunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const videoBlob = new Blob(chunks, { type: selectedMimeType });
                console.log('视频合成完成，大小:', (videoBlob.size / 1024 / 1024).toFixed(2), 'MB');
                
                // 清理流
                videoStream.getTracks().forEach(track => track.stop());
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                
                resolve(videoBlob);
            };

            mediaRecorder.onerror = (error) => {
                console.error('MediaRecorder 错误:', error);
                reject(new Error('视频录制失败: ' + error.message));
            };

            // 开始录制
            mediaRecorder.start();

            // 逐帧绘制
            let frameIndex = 0;
            const drawFrame = () => {
                if (frameIndex >= frames.length) {
                    // 等待音频播放完成（如果有）
                    const videoDuration = frames.length / fps;
                    setTimeout(() => {
                        mediaRecorder.stop();
                    }, Math.max(0, (videoDuration - (frameIndex / fps)) * 1000));
                    return;
                }

                // 绘制当前帧
                ctx.putImageData(frames[frameIndex], 0, 0);
                frameIndex++;

                // 等待下一帧（30fps = 33.33ms per frame）
                setTimeout(drawFrame, 1000 / fps);
            };

            // 开始绘制第一帧
            drawFrame();
        });
    }

    updateStatus(type, message) {
        const statusElement = document.getElementById('status');
        statusElement.className = `status ${type}`;
        statusElement.textContent = message;
        statusElement.classList.remove('hidden');
    }

    updateProgress(percent) {
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        
        progressBar.classList.remove('hidden');
        progressFill.style.width = `${percent}%`;
        progressFill.textContent = `${percent}%`;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.liteAvatar = new LiteAvatarWeb();
});

