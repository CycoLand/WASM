# Browser LLM Chat - Transformers.js POC

A proof-of-concept demonstrating how to run quantized LLMs entirely in the browser using:

- **[Transformers.js](https://huggingface.co/docs/transformers.js)** - Hugging Face's JavaScript ML library
- **[ONNX Runtime Web](https://onnxruntime.ai/)** - Cross-platform ML inference
- **[WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)** - Modern GPU API for the web

## 🚀 Quick Start

### Option 1: Using a Local Server (Recommended)

You need a local HTTP server because ES modules can't be loaded from `file://` URLs.

**Using Python:**
```bash
cd experiments/transformers-js
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

**Using Node.js:**
```bash
npx serve experiments/transformers-js
# Open the URL shown in the terminal
```

**Using VS Code:**
- Install the "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

### Option 2: Using Vite (For Development)

```bash
cd experiments/transformers-js
npm init -y
npm install vite --save-dev
npx vite
```

## 🤖 Available Models

The POC includes multiple models ranked by quality:

| Model | Parameters | Size (Q4) | Quality | Speed | Best For |
|-------|------------|-----------|---------|-------|----------|
| **Llama 3.2 (1B)** | 1B | ~750MB | ⭐⭐⭐⭐ | Fast | Great quality for size |
| **SmolLM2 1.7B** | 1.7B | ~1.0GB | ⭐⭐⭐⭐ | Fast | Good balance |
| **Qwen 2.5 (1.5B)** | 1.5B | ~900MB | ⭐⭐⭐ | Fast | Well optimized |
| **SmolLM2 360M** | 360M | ~250MB | ⭐⭐ | Very Fast | Quick testing |
| **Qwen 2.5 (0.5B)** | 0.5B | ~350MB | ⭐⭐ | Fastest | **Default** - Fast loading |

### Model Recommendations

- **Best Quality**: Llama 3.2 (1B) - Meta's latest, excellent for its size
- **Best Balance**: SmolLM2 1.7B - Fast with good quality
- **Fastest Loading**: Qwen 2.5 (0.5B) - Good for testing the setup

## 📋 Browser Requirements

### Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 113+ | ✅ Full support |
| Edge | 113+ | ✅ Full support |
| Firefox | 141+ | ⚠️ Behind flag (`dom.webgpu.enabled`) |
| Safari | 18+ | ⚠️ Behind flag (Develop menu → Experimental Features → WebGPU) |

### Hardware

- A GPU with WebGPU support (most modern dedicated and integrated GPUs)
- RAM requirements vary by model (1-4GB)
- SSD recommended for faster model caching

## 🎯 Features

- **🔄 Multiple Models** - Choose from 5 different models based on your needs
- **📊 Live Stats** - See tokens/second, inference time, and token count
- **💾 Automatic Caching** - Models are cached after first download
- **📤 Export to Disk** - Save models to your filesystem for backup/sharing
- **📥 Import from Disk** - Load previously exported models without re-downloading
- **🎨 Clean UI** - Modern chat interface with progress indicators
- **🔒 100% Private** - Everything runs locally, data never leaves your device

## 💾 Model Storage & Portability

### Automatic Browser Caching
Models are automatically cached in the browser's Cache Storage after first download. This means:
- Subsequent loads are instant (no re-download)
- Cache persists across browser sessions
- Each origin (domain + port) has isolated storage

### Export Models to Disk 📤
Save your cached models to a folder on your computer:
1. Load a model first (so it's in the cache)
2. Click the **📤 Export** button
3. Select a folder to save to
4. Models are saved with a manifest file for re-importing

**Use cases:**
- Backup your models
- Share models between computers
- Transfer models to another browser profile
- Create an offline model library

### Import Models from Disk 📥
Load previously exported models without downloading:
1. Click the **📥 Import** button
2. Select the folder containing exported models
3. Models are restored to browser cache
4. Load and use immediately!

**Note:** Export/Import requires Chrome or Edge (File System Access API).

## 🔧 Configuration

Edit the `CONFIG` object in `main.js` to customize generation:

```javascript
const CONFIG = {
    dtype: 'q4',           // 'q4', 'q8', 'fp16', 'fp32'
    device: 'webgpu',      // 'webgpu', 'wasm', 'cpu'
    maxNewTokens: 512,     // Max response length
    temperature: 0.7,      // Creativity (0.0 - 1.0)
    topP: 0.9,             // Nucleus sampling threshold
};
```

## 📊 Performance Tips

1. **Choose the right model** - Larger isn't always better for interactive use
2. **Use 4-bit quantization (`q4`)** - Best balance of size and quality
3. **Close other GPU apps** - Frees up GPU memory for inference
4. **Use Chrome/Edge** - Best WebGPU support and performance
5. **First load is slow** - Models are cached after first download (subsequent loads are fast)

## 🐛 Troubleshooting

### "WebGPU is not supported"
- Update your browser to the latest version
- In Firefox: Enable `dom.webgpu.enabled` in `about:config`
- In Safari: Enable WebGPU in Develop → Experimental Features

### "No WebGPU adapter found"
- Your GPU may not support WebGPU
- Try updating your GPU drivers
- Check `chrome://gpu` for WebGPU status

### Model loading fails or is very slow
- Try a smaller model first (Qwen 0.5B)
- Check your internet connection
- Models are large (350MB-2GB) - first download takes time
- Ensure you have enough RAM (check Task Manager/Activity Monitor)

### Generation is slow
- WebGPU performance varies by GPU
- Try a smaller model
- Close other GPU-intensive tabs/applications
- On laptops, ensure you're plugged in (better GPU performance)

### Out of memory errors
- Try a smaller model
- Close other browser tabs
- Restart your browser to free memory

## 📚 Learn More

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [WebGPU Guide](https://huggingface.co/docs/transformers.js/guides/webgpu)
- [Quantization Guide](https://huggingface.co/docs/transformers.js/guides/dtypes)
- [ONNX Community Models](https://huggingface.co/onnx-community)

## 🔗 Model Links

- [Llama-3.2-1B-Instruct](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct)
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct)
- [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)

## 📝 License

MIT - Feel free to use this as a starting point for your own projects!
