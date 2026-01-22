// 国际化文本
const i18nTexts = {
    zh: {
        title: '🎭 LiteAvatar - WASM加速版',
        subtitle: '浏览器版音频到面部动画生成',
        infoTitle: '⚠️ 使用说明：',
        info1: '使用 WASM 后端进行推理（兼容性更好）',
        info2: '首次使用需要下载模型文件（约 100-200MB）',
        info3: '建议使用 16kHz 采样率的 WAV 格式音频',
        info4: '处理时间取决于音频长度和设备性能',
        section0: '0. 预加载模型（可选）',
        preloadDesc: '提前加载模型可以加快后续生成速度',
        preloadBtn: '📦 预加载模型',
        section1: '1. 上传音频文件',
        audioInputText: '点击或拖拽音频文件到这里',
        useDefaultAudio: '使用默认示例音频',
        recordAudio: '🎤 使用麦克风录音',
        stopRecord: '⏹️ 停止录音',
        section2: '2. 加载 Avatar 数据',
        dataInputText: '点击选择 Avatar 数据目录（或使用默认数据）',
        useDefaultData: '使用默认示例数据',
        section3: '3. 生成视频',
        generateBtn: '请先上传音频文件和 Avatar 数据',
        section4: '生成的视频',
        downloadVideo: '下载视频',
        preloadLoading: '正在加载模型，请稍候...',
        preloadSuccess: '✓ 模型加载成功！现在可以开始生成视频了',
        preloadError: '❌ 模型加载失败',
        preloadAlready: '✓ 模型已加载'
    },
    en: {
        title: '🎭 LiteAvatar - WASM Accelerated',
        subtitle: 'Browser-based Audio to Facial Animation',
        infoTitle: '⚠️ Instructions:',
        info1: 'Uses WASM backend for inference (better compatibility)',
        info2: 'First use requires downloading model files (~100-200MB)',
        info3: 'Recommended: 16kHz WAV format audio',
        info4: 'Processing time depends on audio length and device performance',
        section0: '0. Preload Models (Optional)',
        preloadDesc: 'Preloading models can speed up subsequent generation',
        preloadBtn: '📦 Preload Models',
        section1: '1. Upload Audio File',
        audioInputText: 'Click or drag audio file here',
        useDefaultAudio: 'Use Default Sample Audio',
        recordAudio: '🎤 Record with Microphone',
        stopRecord: '⏹️ Stop Recording',
        section2: '2. Load Avatar Data',
        dataInputText: 'Click to select Avatar data directory (or use default data)',
        useDefaultData: 'Use Default Sample Data',
        section3: '3. Generate Video',
        generateBtn: 'Please upload audio file and Avatar data first',
        section4: 'Generated Video',
        downloadVideo: 'Download Video',
        preloadLoading: 'Loading models, please wait...',
        preloadSuccess: '✓ Models loaded successfully! You can now generate videos',
        preloadError: '❌ Model loading failed',
        preloadAlready: '✓ Models already loaded'
    }
};

// 当前语言
let currentLang = localStorage.getItem('liteAvatarLang') || 'zh';

// 切换语言
function switchLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('liteAvatarLang', lang);
    
    // 更新所有带 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (i18nTexts[lang] && i18nTexts[lang][key]) {
            element.textContent = i18nTexts[lang][key];
        }
    });
    
    // 更新语言按钮状态
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.getAttribute('data-lang') === lang) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // 更新 HTML lang 属性
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
}

// 获取翻译文本
function t(key) {
    return i18nTexts[currentLang] && i18nTexts[currentLang][key] ? i18nTexts[currentLang][key] : key;
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 设置初始语言
    switchLanguage(currentLang);
    
    // 绑定语言切换按钮
    document.getElementById('langZh').addEventListener('click', () => switchLanguage('zh'));
    document.getElementById('langEn').addEventListener('click', () => switchLanguage('en'));
});

// 导出函数供其他脚本使用
window.i18n = {
    switchLanguage,
    t,
    currentLang: () => currentLang
};
