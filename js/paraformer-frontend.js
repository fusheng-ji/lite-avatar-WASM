/**
 * Paraformer 前端特征提取模块
 * 实现 fbank + LFR + CMVN，与后端 Python 实现保持一致
 */

class ParaformerFrontend {
    constructor(config = {}) {
        this.sampleRate = config.sampleRate || 16000;
        this.nMels = config.nMels || 80;
        this.frameLength = config.frameLength || 400;  // 25ms @ 16kHz
        this.frameShift = config.frameShift || 160;   // 10ms @ 16kHz
        this.windowType = config.windowType || 'hamming';
        this.lfrM = config.lfrM || 7;
        this.lfrN = config.lfrN || 6;
        
        // CMVN 统计量（需要从后端获取或使用默认值）
        this.cmvnMean = config.cmvnMean || null;
        this.cmvnStd = config.cmvnStd || null;
        
        // 预计算 Mel 滤波器组
        this.melFilterBank = this.createMelFilterBank();
    }

    /**
     * 创建 Mel 滤波器组
     */
    createMelFilterBank() {
        const nFft = 512; // FFT 点数
        const fMin = 0;
        const fMax = this.sampleRate / 2;
        
        // Mel 频率范围
        const melMin = this.hzToMel(fMin);
        const melMax = this.hzToMel(fMax);
        const melPoints = Array.from({ length: this.nMels + 2 }, (_, i) => {
            return melMin + (melMax - melMin) * i / (this.nMels + 1);
        });
        
        // 转换为 Hz
        const hzPoints = melPoints.map(m => this.melToHz(m));
        
        // 转换为 FFT bin
        const fftBins = hzPoints.map(hz => Math.floor((hz / this.sampleRate) * nFft));
        
        // 创建滤波器组
        const filterBank = [];
        for (let i = 0; i < this.nMels; i++) {
            const filter = new Float32Array(nFft / 2 + 1);
            const start = fftBins[i];
            const peak = fftBins[i + 1];
            const end = fftBins[i + 2];
            
            // 上升沿
            for (let k = start; k < peak; k++) {
                if (k >= 0 && k < filter.length) {
                    filter[k] = (k - start) / (peak - start);
                }
            }
            
            // 下降沿
            for (let k = peak; k < end; k++) {
                if (k >= 0 && k < filter.length) {
                    filter[k] = (end - k) / (end - peak);
                }
            }
            
            filterBank.push(filter);
        }
        
        return filterBank;
    }

    hzToMel(hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    melToHz(mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }

    /**
     * 应用 Hamming 窗
     */
    applyWindow(frame) {
        const window = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (frame.length - 1));
        }
        return frame.map((val, i) => val * window[i]);
    }

    /**
     * FFT（迭代实现，更高效）
     */
    fft(signal) {
        const n = signal.length;
        
        // 确保 n 是 2 的幂
        const nextPow2 = Math.pow(2, Math.ceil(Math.log2(n)));
        const padded = new Float32Array(nextPow2);
        padded.set(signal);
        
        // 位反转
        const rev = new Array(nextPow2);
        for (let i = 0; i < nextPow2; i++) {
            rev[i] = 0;
            let j = i;
            for (let k = 0; k < Math.log2(nextPow2); k++) {
                rev[i] = (rev[i] << 1) | (j & 1);
                j >>= 1;
            }
        }
        
        // 迭代 FFT
        const output = new Float32Array(nextPow2 * 2); // [real, imag, real, imag, ...]
        for (let i = 0; i < nextPow2; i++) {
            output[i * 2] = padded[rev[i]];
            output[i * 2 + 1] = 0;
        }
        
        for (let len = 2; len <= nextPow2; len <<= 1) {
            const angle = -2 * Math.PI / len;
            for (let i = 0; i < nextPow2; i += len) {
                for (let j = 0; j < len / 2; j++) {
                    const u = i + j;
                    const v = i + j + len / 2;
                    const tReal = output[u * 2];
                    const tImag = output[u * 2 + 1];
                    const wReal = Math.cos(angle * j);
                    const wImag = Math.sin(angle * j);
                    const vReal = output[v * 2];
                    const vImag = output[v * 2 + 1];
                    
                    output[u * 2] = tReal + wReal * vReal - wImag * vImag;
                    output[u * 2 + 1] = tImag + wReal * vImag + wImag * vReal;
                    output[v * 2] = tReal - (wReal * vReal - wImag * vImag);
                    output[v * 2 + 1] = tImag - (wReal * vImag + wImag * vReal);
                }
            }
        }
        
        return output;
    }

    /**
     * 计算功率谱
     */
    powerSpectrum(fftResult) {
        // fftResult 格式：[real0, imag0, real1, imag1, ...]
        const n = fftResult.length / 2;
        const power = new Float32Array(n + 1);
        
        // DC 分量
        power[0] = Math.pow(fftResult[0], 2);
        
        // 频率分量
        for (let i = 1; i < n; i++) {
            const real = fftResult[i * 2];
            const imag = fftResult[i * 2 + 1];
            power[i] = real * real + imag * imag;
        }
        
        // Nyquist 频率
        power[n] = Math.pow(fftResult[n * 2], 2);
        
        return power;
    }

    /**
     * 提取 fbank 特征
     */
    extractFbank(waveform) {
        const frames = [];
        const numFrames = Math.floor((waveform.length - this.frameLength) / this.frameShift) + 1;
        
        for (let i = 0; i < numFrames; i++) {
            const start = i * this.frameShift;
            const end = start + this.frameLength;
            const frame = waveform.slice(start, end);
            
            // 应用窗函数
            const windowed = this.applyWindow(frame);
            
            // 零填充到 512
            const padded = new Float32Array(512);
            padded.set(windowed);
            
            // FFT
            const fftResult = this.fft(padded);
            
            // 功率谱
            const power = this.powerSpectrum(fftResult);
            
            // Mel 滤波器组
            const melFeatures = new Float32Array(this.nMels);
            for (let m = 0; m < this.nMels; m++) {
                let sum = 0;
                for (let k = 0; k < power.length; k++) {
                    sum += power[k] * this.melFilterBank[m][k];
                }
                melFeatures[m] = Math.log(Math.max(sum, 1e-10));
            }
            
            frames.push(melFeatures);
        }
        
        return frames;
    }

    /**
     * 应用 LFR (Low Frame Rate)
     */
    applyLFR(fbankFeatures) {
        const numFrames = fbankFeatures.length;
        const numLfrFrames = Math.floor((numFrames - this.lfrM) / this.lfrN) + 1;
        const lfrFeatures = [];
        
        for (let i = 0; i < numLfrFrames; i++) {
            const startIdx = i * this.lfrN;
            const endIdx = Math.min(startIdx + this.lfrM, numFrames);
            
            // 拼接 M 帧
            const concatFrame = new Float32Array(this.nMels * this.lfrM);
            for (let j = 0; j < this.lfrM; j++) {
                const frameIdx = startIdx + j;
                if (frameIdx < numFrames) {
                    concatFrame.set(fbankFeatures[frameIdx], j * this.nMels);
                }
            }
            
            lfrFeatures.push(concatFrame);
        }
        
        return lfrFeatures;
    }

    /**
     * 应用 CMVN (Cepstral Mean and Variance Normalization)
     */
    applyCMVN(features) {
        if (!this.cmvnMean || !this.cmvnStd) {
            // 如果没有提供 CMVN，计算当前特征的统计量
            const numFrames = features.length;
            const featDim = features[0].length;
            const mean = new Float32Array(featDim);
            const std = new Float32Array(featDim);
            
            // 计算均值
            for (let i = 0; i < numFrames; i++) {
                for (let d = 0; d < featDim; d++) {
                    mean[d] += features[i][d];
                }
            }
            for (let d = 0; d < featDim; d++) {
                mean[d] /= numFrames;
            }
            
            // 计算标准差
            for (let i = 0; i < numFrames; i++) {
                for (let d = 0; d < featDim; d++) {
                    std[d] += Math.pow(features[i][d] - mean[d], 2);
                }
            }
            for (let d = 0; d < featDim; d++) {
                std[d] = Math.sqrt(std[d] / numFrames);
            }
            
            this.cmvnMean = mean;
            this.cmvnStd = std;
        }
        
        // 应用归一化
        const normalized = [];
        for (let i = 0; i < features.length; i++) {
            const normFrame = new Float32Array(features[i].length);
            for (let d = 0; d < features[i].length; d++) {
                normFrame[d] = (features[i][d] - this.cmvnMean[d]) / (this.cmvnStd[d] + 1e-8);
            }
            normalized.push(normFrame);
        }
        
        return normalized;
    }

    /**
     * 完整的前端处理流程
     */
    process(waveform) {
        // 1. fbank
        const fbankFeatures = this.extractFbank(waveform);
        
        // 2. LFR
        const lfrFeatures = this.applyLFR(fbankFeatures);
        
        // 3. CMVN
        const cmvnFeatures = this.applyCMVN(lfrFeatures);
        
        // 转换为 [T, D] 格式的 Float32Array
        const numFrames = cmvnFeatures.length;
        const featDim = cmvnFeatures[0].length;
        const output = new Float32Array(numFrames * featDim);
        
        for (let i = 0; i < numFrames; i++) {
            output.set(cmvnFeatures[i], i * featDim);
        }
        
        return {
            features: output,
            numFrames: numFrames,
            featDim: featDim
        };
    }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ParaformerFrontend;
} else {
    window.ParaformerFrontend = ParaformerFrontend;
}
