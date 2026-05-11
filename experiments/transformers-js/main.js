/**
 * Browser LLM Chat - Transformers.js POC
 * 
 * This demonstrates running quantized LLMs in the browser using:
 * - Transformers.js v3 (Hugging Face)
 * - ONNX Runtime Web
 * - WebGPU acceleration
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

// ============================================================================
// Cache Configuration
// ============================================================================

// Transformers.js uses the browser's Cache API by default
// Models are cached in: Cache Storage → "transformers-cache"
// This persists across sessions - no re-download needed!
env.useBrowserCache = true;       // Enable browser cache (default: true)
env.allowLocalModels = false;     // We're loading from HuggingFace Hub

// ============================================================================
// Available Models - Ranked by quality (best first)
// ============================================================================

// NOTE: These are verified working models from onnx-community that support
// text-generation pipeline with WebGPU. Model availability may change.
// Check https://huggingface.co/onnx-community for latest models.

const MODELS = {
    'llama-3.2-1b': {
        id: 'onnx-community/Llama-3.2-1B-Instruct',
        name: 'Llama 3.2 (1B)',
        description: 'Meta\'s latest small model - great quality for size',
        size: '~750MB',
        quality: 4,
        speed: 4,
    },
    'smollm2-1.7b': {
        id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        name: 'SmolLM2 1.7B',
        description: 'Great balance of quality and speed',
        size: '~1.0GB',
        quality: 4,
        speed: 4,
    },
    'qwen2.5-1.5b': {
        id: 'onnx-community/Qwen2.5-1.5B-Instruct',
        name: 'Qwen 2.5 (1.5B)',
        description: 'Alibaba\'s model - good quality, well optimized',
        size: '~900MB',
        quality: 3,
        speed: 4,
    },
    'smollm2-360m': {
        id: 'HuggingFaceTB/SmolLM2-360M-Instruct',
        name: 'SmolLM2 360M',
        description: 'Fast and small - good for testing',
        size: '~250MB',
        quality: 2,
        speed: 5,
    },
    'qwen2.5-0.5b': {
        id: 'onnx-community/Qwen2.5-0.5B-Instruct',
        name: 'Qwen 2.5 (0.5B)',
        description: 'Fastest loading, basic capabilities',
        size: '~350MB',
        quality: 2,
        speed: 5,
    },
};

// Default to Qwen 0.5B for fast testing, switch to larger models for better quality
const DEFAULT_MODEL = 'qwen2.5-0.5b';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    dtype: 'q4',           // 4-bit quantization for smaller size
    device: 'webgpu',      // Use WebGPU for GPU acceleration
    maxNewTokens: 512,     // Maximum tokens to generate
    temperature: 0.7,      // Sampling temperature
    topP: 0.9,             // Nucleus sampling
    doSample: true,        // Enable sampling (vs greedy)
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    modelSelect: document.getElementById('modelSelect'),
    modelInfo: document.getElementById('modelInfo'),
    loadModelBtn: document.getElementById('loadModelBtn'),
    clearCacheBtn: document.getElementById('clearCacheBtn'),
    exportModelsBtn: document.getElementById('exportModelsBtn'),
    importModelsBtn: document.getElementById('importModelsBtn'),
    cacheStatus: document.getElementById('cacheStatus'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    progressBar: document.getElementById('progressBar'),
    chatContainer: document.getElementById('chatContainer'),
    userInput: document.getElementById('userInput'),
    sendButton: document.getElementById('sendButton'),
    stats: document.getElementById('stats'),
    tokenCount: document.getElementById('tokenCount'),
    tokensPerSec: document.getElementById('tokensPerSec'),
    inferenceTime: document.getElementById('inferenceTime'),
};

// ============================================================================
// State
// ============================================================================

let generator = null;
let isGenerating = false;
let conversationHistory = [];
let currentModelKey = null;

// ============================================================================
// Cache Management
// ============================================================================

const CACHE_NAME = 'transformers-cache';

async function getCacheInfo() {
    try {
        const cache = await caches.open(CACHE_NAME);
        const keys = await cache.keys();
        
        let totalSize = 0;
        const cachedModels = new Set();
        
        for (const request of keys) {
            // Try to get the size from the response
            const response = await cache.match(request);
            if (response) {
                const blob = await response.clone().blob();
                totalSize += blob.size;
                
                // Extract model name from URL
                const url = request.url;
                for (const [key, model] of Object.entries(MODELS)) {
                    if (url.includes(model.id.replace('/', '%2F')) || url.includes(model.id)) {
                        cachedModels.add(key);
                    }
                }
            }
        }
        
        return {
            totalSize,
            totalSizeMB: (totalSize / 1024 / 1024).toFixed(1),
            fileCount: keys.length,
            cachedModels: Array.from(cachedModels),
        };
    } catch (error) {
        console.warn('Could not get cache info:', error);
        return { totalSize: 0, totalSizeMB: '0', fileCount: 0, cachedModels: [] };
    }
}

async function clearCache() {
    try {
        const deleted = await caches.delete(CACHE_NAME);
        console.log('Cache cleared:', deleted);
        return deleted;
    } catch (error) {
        console.error('Error clearing cache:', error);
        return false;
    }
}

async function updateCacheStatus() {
    const info = await getCacheInfo();
    
    if (info.fileCount > 0) {
        const modelNames = info.cachedModels.map(key => MODELS[key]?.name || key).join(', ');
        elements.cacheStatus.innerHTML = `
            <span class="cache-icon">💾</span>
            <span class="cache-text">
                <strong>${info.totalSizeMB} MB</strong> cached
                ${info.cachedModels.length > 0 ? `<br><small>Models: ${modelNames}</small>` : ''}
            </span>
        `;
        elements.clearCacheBtn.style.display = 'inline-block';
    } else {
        elements.cacheStatus.innerHTML = `
            <span class="cache-icon">📭</span>
            <span class="cache-text">No models cached yet</span>
        `;
        elements.clearCacheBtn.style.display = 'none';
    }
}

async function handleClearCache() {
    if (!confirm('Clear all cached models? You will need to re-download them next time.')) {
        return;
    }
    
    elements.clearCacheBtn.disabled = true;
    elements.clearCacheBtn.textContent = 'Clearing...';
    
    await clearCache();
    
    // Also clear IndexedDB which ONNX Runtime uses
    try {
        const databases = await indexedDB.databases();
        for (const db of databases) {
            if (db.name && (db.name.includes('onnx') || db.name.includes('transformers'))) {
                indexedDB.deleteDatabase(db.name);
            }
        }
    } catch (e) {
        // indexedDB.databases() not supported in all browsers
    }
    
    elements.clearCacheBtn.textContent = 'Clear Cache';
    elements.clearCacheBtn.disabled = false;
    
    await updateCacheStatus();
    
    addMessage('assistant', '🗑️ Cache cleared! Models will be downloaded fresh on next load.');
}

// ============================================================================
// File System Access API - Save/Load Models to Disk
// ============================================================================

// Check if File System Access API is supported
function isFileSystemAccessSupported() {
    return 'showDirectoryPicker' in window;
}

/**
 * Export cached models to a user-selected folder
 * This allows users to save models to disk for:
 * - Backup
 * - Sharing between browsers/computers
 * - Offline use without re-downloading
 */
async function exportModelsToFolder() {
    if (!isFileSystemAccessSupported()) {
        alert('File System Access API is not supported in this browser.\nPlease use Chrome or Edge.');
        return;
    }
    
    try {
        // Ask user to select a folder
        const dirHandle = await window.showDirectoryPicker({
            mode: 'readwrite',
            startIn: 'downloads',
        });
        
        updateStatus('Exporting models to folder...', 'loading');
        
        const cache = await caches.open(CACHE_NAME);
        const keys = await cache.keys();
        
        if (keys.length === 0) {
            alert('No models cached to export. Load a model first!');
            updateStatus('No models to export', 'error');
            return;
        }
        
        let exportedCount = 0;
        let exportedSize = 0;
        
        // Create a manifest to track what we're exporting
        const manifest = {
            version: 1,
            exportedAt: new Date().toISOString(),
            files: [],
        };
        
        for (const request of keys) {
            const response = await cache.match(request);
            if (!response) continue;
            
            const blob = await response.blob();
            const url = new URL(request.url);
            
            // Create a safe filename from the URL path
            // e.g., "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx"
            // becomes "onnx-community__Qwen2.5-0.5B-Instruct__onnx__model_q4.onnx"
            const safeName = url.pathname
                .replace(/^\//, '')  // Remove leading slash
                .replace(/\//g, '__'); // Replace slashes with double underscore
            
            try {
                const fileHandle = await dirHandle.getFileHandle(safeName, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(blob);
                await writable.close();
                
                manifest.files.push({
                    originalUrl: request.url,
                    fileName: safeName,
                    size: blob.size,
                    contentType: response.headers.get('content-type') || 'application/octet-stream',
                });
                
                exportedCount++;
                exportedSize += blob.size;
                
                updateStatus(`Exporting: ${exportedCount}/${keys.length} files...`, 'loading');
            } catch (fileError) {
                console.warn(`Failed to export ${safeName}:`, fileError);
            }
        }
        
        // Save the manifest
        const manifestHandle = await dirHandle.getFileHandle('transformers-manifest.json', { create: true });
        const manifestWritable = await manifestHandle.createWritable();
        await manifestWritable.write(JSON.stringify(manifest, null, 2));
        await manifestWritable.close();
        
        const sizeMB = (exportedSize / 1024 / 1024).toFixed(1);
        updateStatus(`✅ Exported ${exportedCount} files (${sizeMB} MB)`, 'ready');
        
        addMessage('assistant', 
            `📁 Models exported successfully!\n\n` +
            `• Files: ${exportedCount}\n` +
            `• Size: ${sizeMB} MB\n` +
            `• Location: ${dirHandle.name}/\n\n` +
            `You can now copy this folder to another computer or browser profile and import it.`
        );
        
    } catch (error) {
        if (error.name === 'AbortError') {
            updateStatus('Export cancelled', 'ready');
            return;
        }
        console.error('Export error:', error);
        updateStatus(`Export failed: ${error.message}`, 'error');
    }
}

/**
 * Import models from a previously exported folder
 * This restores models to the browser cache without re-downloading
 */
async function importModelsFromFolder() {
    if (!isFileSystemAccessSupported()) {
        alert('File System Access API is not supported in this browser.\nPlease use Chrome or Edge.');
        return;
    }
    
    try {
        // Ask user to select the folder containing exported models
        const dirHandle = await window.showDirectoryPicker({
            mode: 'read',
            startIn: 'downloads',
        });
        
        updateStatus('Reading manifest...', 'loading');
        
        // Try to read the manifest
        let manifest;
        try {
            const manifestHandle = await dirHandle.getFileHandle('transformers-manifest.json');
            const manifestFile = await manifestHandle.getFile();
            manifest = JSON.parse(await manifestFile.text());
        } catch (e) {
            alert('Could not find transformers-manifest.json in the selected folder.\nMake sure you select a folder that was previously exported.');
            updateStatus('Import failed: No manifest found', 'error');
            return;
        }
        
        updateStatus(`Importing ${manifest.files.length} files...`, 'loading');
        
        const cache = await caches.open(CACHE_NAME);
        let importedCount = 0;
        let importedSize = 0;
        
        for (const fileInfo of manifest.files) {
            try {
                const fileHandle = await dirHandle.getFileHandle(fileInfo.fileName);
                const file = await fileHandle.getFile();
                
                // Create a Response object to store in cache
                const response = new Response(file, {
                    headers: {
                        'content-type': fileInfo.contentType,
                        'content-length': fileInfo.size.toString(),
                    },
                });
                
                // Store in cache with original URL as key
                await cache.put(fileInfo.originalUrl, response);
                
                importedCount++;
                importedSize += fileInfo.size;
                
                updateStatus(`Importing: ${importedCount}/${manifest.files.length} files...`, 'loading');
            } catch (fileError) {
                console.warn(`Failed to import ${fileInfo.fileName}:`, fileError);
            }
        }
        
        await updateCacheStatus();
        
        const sizeMB = (importedSize / 1024 / 1024).toFixed(1);
        updateStatus(`✅ Imported ${importedCount} files (${sizeMB} MB)`, 'ready');
        
        addMessage('assistant', 
            `📥 Models imported successfully!\n\n` +
            `• Files: ${importedCount}\n` +
            `• Size: ${sizeMB} MB\n\n` +
            `The models are now in your browser cache. Select one and click "Load Model" to use it!`
        );
        
    } catch (error) {
        if (error.name === 'AbortError') {
            updateStatus('Import cancelled', 'ready');
            return;
        }
        console.error('Import error:', error);
        updateStatus(`Import failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// UI Helpers
// ============================================================================

function populateModelSelect() {
    elements.modelSelect.innerHTML = '';
    
    for (const [key, model] of Object.entries(MODELS)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = `${model.name} (${model.size})`;
        if (key === DEFAULT_MODEL) {
            option.selected = true;
        }
        elements.modelSelect.appendChild(option);
    }
    
    updateModelInfo();
}

function updateModelInfo() {
    const model = MODELS[elements.modelSelect.value];
    const qualityStars = '⭐'.repeat(model.quality) + '☆'.repeat(5 - model.quality);
    const speedStars = '🚀'.repeat(model.speed) + '·'.repeat(5 - model.speed);
    
    elements.modelInfo.innerHTML = `
        <strong>${model.name}</strong><br>
        ${model.description}<br>
        <span style="font-size: 0.85em; color: #9ca3af;">
            Quality: ${qualityStars} | Speed: ${speedStars} | Size: ${model.size}
        </span>
    `;
}

function updateStatus(text, state = 'loading') {
    elements.statusText.textContent = text;
    elements.statusIndicator.className = 'indicator';
    if (state === 'ready') {
        elements.statusIndicator.classList.add('ready');
    } else if (state === 'error') {
        elements.statusIndicator.classList.add('error');
    }
}

function updateProgress(progress) {
    elements.progressBar.style.width = `${progress}%`;
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = content;
    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    return messageDiv;
}

function updateStats(tokens, timeMs) {
    elements.stats.style.display = 'flex';
    elements.tokenCount.textContent = tokens;
    elements.inferenceTime.textContent = timeMs.toFixed(0);
    elements.tokensPerSec.textContent = timeMs > 0 ? (tokens / (timeMs / 1000)).toFixed(1) : '0';
}

function enableInput(enabled) {
    elements.userInput.disabled = !enabled;
    elements.sendButton.disabled = !enabled;
    if (enabled) {
        elements.userInput.focus();
    }
}

function enableModelSelection(enabled) {
    elements.modelSelect.disabled = !enabled;
    elements.loadModelBtn.disabled = !enabled;
}

// ============================================================================
// WebGPU Check
// ============================================================================

async function checkWebGPU() {
    if (!navigator.gpu) {
        throw new Error(
            'WebGPU is not supported in this browser. ' +
            'Please use Chrome 113+, Edge 113+, or enable WebGPU in your browser settings.'
        );
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error(
            'No WebGPU adapter found. Your GPU may not be supported.'
        );
    }

    const adapterInfo = await adapter.requestAdapterInfo?.() || {};
    console.log('WebGPU Adapter:', adapterInfo);
    
    return true;
}

// ============================================================================
// Model Loading
// ============================================================================

async function loadModel() {
    const modelKey = elements.modelSelect.value;
    const modelConfig = MODELS[modelKey];
    
    if (currentModelKey === modelKey && generator) {
        updateStatus(`${modelConfig.name} already loaded!`, 'ready');
        return;
    }
    
    // Clear previous model
    generator = null;
    currentModelKey = null;
    conversationHistory = [];
    elements.chatContainer.innerHTML = '';
    
    enableModelSelection(false);
    enableInput(false);
    updateProgress(0);
    
    // Check if model might be cached
    const cacheInfo = await getCacheInfo();
    const isCached = cacheInfo.cachedModels.includes(modelKey);
    
    try {
        updateStatus('Checking WebGPU support...', 'loading');
        await checkWebGPU();
        
        if (isCached) {
            updateStatus(`Loading ${modelConfig.name} from cache... ⚡`, 'loading');
        } else {
            updateStatus(`Downloading ${modelConfig.name}... (first time only, will be cached)`, 'loading');
        }
        
        let lastProgress = 0;
        let downloadedFiles = new Set();
        let isFromCache = true; // Assume cache until we see download progress
        
        generator = await pipeline('text-generation', modelConfig.id, {
            dtype: CONFIG.dtype,
            device: CONFIG.device,
            progress_callback: (progress) => {
                if (progress.status === 'downloading' || progress.status === 'progress') {
                    isFromCache = false; // We're actually downloading
                    downloadedFiles.add(progress.file);
                    const fileProgress = progress.progress || (progress.loaded / progress.total * 100);
                    
                    if (fileProgress > lastProgress || downloadedFiles.size > 1) {
                        lastProgress = fileProgress;
                        updateProgress(Math.min(fileProgress, 99));
                    }
                    
                    const sizeMB = progress.total ? (progress.total / 1024 / 1024).toFixed(0) : '?';
                    const loadedMB = progress.loaded ? (progress.loaded / 1024 / 1024).toFixed(1) : '0';
                    updateStatus(
                        `Downloading: ${progress.file?.split('/').pop() || 'model'} (${loadedMB}/${sizeMB}MB)`,
                        'loading'
                    );
                } else if (progress.status === 'loading') {
                    updateStatus(`Loading ${modelConfig.name} into memory...`, 'loading');
                } else if (progress.status === 'ready') {
                    // File was loaded from cache
                }
            },
        });

        currentModelKey = modelKey;
        updateProgress(100);
        
        // Update cache status after loading
        await updateCacheStatus();
        
        const loadSource = isFromCache ? '(from cache ⚡)' : '(now cached for next time 💾)';
        updateStatus(`✅ ${modelConfig.name} ready ${loadSource}`, 'ready');
        enableInput(true);
        
        console.log('Model loaded successfully!');
        console.log('Model:', modelConfig.id);
        console.log('Configuration:', CONFIG);
        console.log('Loaded from cache:', isFromCache);
        
    } catch (error) {
        console.error('Model loading error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        
        addMessage(
            'assistant',
            `⚠️ Failed to load model: ${error.message}\n\n` +
            'Tips:\n' +
            '• Make sure you\'re using Chrome 113+ or Edge 113+\n' +
            '• Check that WebGPU is enabled in your browser\n' +
            '• Try a smaller model if you\'re running low on memory\n' +
            '• Check the browser console for more details'
        );
    } finally {
        enableModelSelection(true);
    }
}

// ============================================================================
// Chat Logic
// ============================================================================

async function generateResponse(userMessage) {
    if (isGenerating || !generator) return;
    
    isGenerating = true;
    enableInput(false);
    
    // Add user message to UI and history
    addMessage('user', userMessage);
    conversationHistory.push({
        role: 'user',
        content: userMessage,
    });

    // Create assistant message placeholder
    const assistantDiv = addMessage('assistant', '');
    assistantDiv.classList.add('generating');

    try {
        const startTime = performance.now();
        let generatedText = '';
        let tokenCount = 0;

        // Build the messages array for the model
        const messages = [
            {
                role: 'system',
                content: 'You are a helpful, friendly AI assistant. Provide clear, informative responses.',
            },
            ...conversationHistory,
        ];

        // Generate response
        const output = await generator(messages, {
            max_new_tokens: CONFIG.maxNewTokens,
            temperature: CONFIG.temperature,
            top_p: CONFIG.topP,
            do_sample: CONFIG.doSample,
            return_full_text: false,
            callback_function: (output) => {
                if (output && output[0] && output[0].generated_text) {
                    const newText = output[0].generated_text;
                    if (typeof newText === 'string') {
                        generatedText = newText;
                    } else if (Array.isArray(newText) && newText.length > 0) {
                        const lastMessage = newText[newText.length - 1];
                        if (lastMessage && lastMessage.content) {
                            generatedText = lastMessage.content;
                        }
                    }
                    assistantDiv.textContent = generatedText;
                    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
                    tokenCount++;
                }
            },
        });

        // Extract final response
        let finalResponse = '';
        if (output && output[0]) {
            const generated = output[0].generated_text;
            if (typeof generated === 'string') {
                finalResponse = generated;
            } else if (Array.isArray(generated) && generated.length > 0) {
                const lastMessage = generated[generated.length - 1];
                if (lastMessage && lastMessage.content) {
                    finalResponse = lastMessage.content;
                }
            }
        }

        const endTime = performance.now();
        
        // Update UI with final response
        assistantDiv.classList.remove('generating');
        assistantDiv.textContent = finalResponse || 'Sorry, I could not generate a response.';
        
        // Add to conversation history
        conversationHistory.push({
            role: 'assistant',
            content: finalResponse,
        });

        // Update stats (estimate token count from characters)
        const estimatedTokens = Math.ceil(finalResponse.length / 4);
        updateStats(estimatedTokens, endTime - startTime);

    } catch (error) {
        console.error('Generation error:', error);
        assistantDiv.classList.remove('generating');
        assistantDiv.textContent = `Error: ${error.message}`;
        assistantDiv.style.color = '#f87171';
    } finally {
        isGenerating = false;
        enableInput(true);
    }
}

// ============================================================================
// Event Handlers
// ============================================================================

elements.modelSelect.addEventListener('change', updateModelInfo);

elements.loadModelBtn.addEventListener('click', loadModel);

elements.sendButton.addEventListener('click', () => {
    const message = elements.userInput.value.trim();
    if (message) {
        elements.userInput.value = '';
        generateResponse(message);
    }
});

elements.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.sendButton.click();
    }
});

elements.clearCacheBtn.addEventListener('click', handleClearCache);

elements.exportModelsBtn.addEventListener('click', exportModelsToFolder);

elements.importModelsBtn.addEventListener('click', importModelsFromFolder);

// ============================================================================
// Initialize
// ============================================================================

async function init() {
    populateModelSelect();
    updateStatus('Select a model and click "Load Model" to begin.', 'loading');
    elements.statusIndicator.classList.remove('ready');
    
    // Check and display cache status
    await updateCacheStatus();
    
    // Add welcome message
    const cacheInfo = await getCacheInfo();
    let welcomeMsg = '👋 Welcome! Select a model from the dropdown above and click "Load Model" to get started.\n\n';
    
    if (cacheInfo.cachedModels.length > 0) {
        welcomeMsg += `✨ Good news! You have ${cacheInfo.cachedModels.length} model(s) cached (${cacheInfo.totalSizeMB} MB). ` +
            'These will load instantly without downloading again!\n\n';
    }
    
    welcomeMsg += '💡 Tip: Models are automatically cached after first download. ' +
        'Subsequent loads are much faster!';
    
    addMessage('assistant', welcomeMsg);
}

// Start the app
init();
