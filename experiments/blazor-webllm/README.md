# Blazor WebLLM - Browser-Based LLM Chat

A Blazor WebAssembly application that runs LLM inference entirely in the browser using [WebLLM](https://webllm.mlc.ai/) and WebGPU.

## 🚀 Features

- **100% Client-Side**: Everything runs in the browser - no server required
- **WebGPU Acceleration**: Fast inference using your GPU
- **Microsoft.Extensions.AI Compatible**: Uses the standard `IChatClient` interface
- **Multiple Models**: Choose from Llama 3.2, Phi 3.5, Qwen 2.5, SmolLM2
- **Streaming Support**: Real-time token-by-token response display
- **Model Caching**: Models cached in browser storage after first download
- **Export/Import**: Save models to disk for backup and portability
- **Privacy**: Your data never leaves your device

## 📋 Requirements

### Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 113+ | ✅ Full support |
| Edge | 113+ | ✅ Full support |
| Firefox | 141+ | ⚠️ Behind flag |
| Safari | 18+ | ⚠️ Behind flag |

### Hardware

- GPU with WebGPU support
- 2-4GB RAM depending on model size

## 🛠️ Quick Start

```bash
cd experiments/blazor-webllm
dotnet run
```

Then open https://localhost:5000 (or the URL shown in terminal).

## 📁 Project Structure

```
experiments/blazor-webllm/
├── Components/
│   ├── ModelManager.razor      # Model selection, loading, cache management
│   └── ChatInterface.razor     # Chat UI with streaming support
├── Services/
│   ├── IWebLLMServices.cs      # Service interfaces
│   ├── Models.cs               # Data models (ModelInfo, CacheInfo, etc.)
│   ├── WebLLMModelManager.cs   # Model management implementation
│   └── WebLLMChatService.cs    # IChatClient implementation
├── wwwroot/
│   └── js/
│       └── webllm-interop.js   # JavaScript interop for WebLLM
├── Pages/
│   └── Home.razor              # Main page
└── Program.cs                  # Service registration
```

## 🤖 Available Models

| Model | Parameters | Size | Quality | Speed |
|-------|------------|------|---------|-------|
| Llama 3.2 (1B) | 1B | ~750MB | ⭐⭐⭐⭐ | Fast |
| SmolLM2 1.7B | 1.7B | ~1.0GB | ⭐⭐⭐⭐ | Fast |
| Qwen 2.5 (1.5B) | 1.5B | ~900MB | ⭐⭐⭐ | Fast |
| Qwen 2.5 (0.5B) | 0.5B | ~350MB | ⭐⭐ | Fastest |
| Phi 3.5 Mini | 3.8B | ~2.1GB | ⭐⭐⭐⭐⭐ | Slower |

## 💻 Using the Services

### Basic Usage (Non-Streaming)

```csharp
@inject IWebLLMChatService ChatService
@inject IModelManagerService ModelManager

// Load a model first
await ModelManager.LoadModelAsync("Llama-3.2-1B-Instruct-q4f16_1-MLC");

// Create chat messages
var messages = new List<ChatMessage>
{
    new(ChatRole.System, "You are a helpful assistant."),
    new(ChatRole.User, "What is WebGPU?")
};

// Get completion (implements Microsoft.Extensions.AI.IChatClient)
var response = await ChatService.CompleteAsync(messages);
Console.WriteLine(response.Message.Text);
```

### Streaming Usage

```csharp
await foreach (var update in ChatService.CompleteStreamingAsync(messages))
{
    foreach (var content in update.Contents.OfType<TextContent>())
    {
        Console.Write(content.Text); // Tokens appear as generated
    }
}
```

### Model Management

```csharp
// Get available models
var models = ModelManager.AvailableModels;

// Check cache status
var cacheInfo = await ModelManager.GetCacheInfoAsync();
Console.WriteLine($"Cached: {cacheInfo.TotalSizeMB} MB");

// Export models to folder
var result = await ModelManager.ExportModelsToFolderAsync();

// Import models from folder
var result = await ModelManager.ImportModelsFromFolderAsync();

// Clear cache
await ModelManager.ClearCacheAsync();
```

## 🔌 Integration with Your App

### 1. Copy the Services

Copy these files to your project:
- `Services/IWebLLMServices.cs`
- `Services/Models.cs`
- `Services/WebLLMModelManager.cs`
- `Services/WebLLMChatService.cs`
- `wwwroot/js/webllm-interop.js`

### 2. Register Services

```csharp
// In Program.cs
builder.Services.AddSingleton<IModelManagerService, WebLLMModelManager>();
builder.Services.AddSingleton<IWebLLMChatService, WebLLMChatService>();
```

### 3. Use IChatClient

Since `IWebLLMChatService` implements `Microsoft.Extensions.AI.IChatClient`, you can use it as a drop-in replacement:

```csharp
// Your existing code using IChatClient
public class MyService
{
    private readonly IChatClient _chatClient;

    public MyService(IWebLLMChatService chatService)
    {
        _chatClient = chatService; // Works because it implements IChatClient
    }

    public async Task<string> GetResponseAsync(string prompt)
    {
        var response = await _chatClient.CompleteAsync([
            new ChatMessage(ChatRole.User, prompt)
        ]);
        return response.Message.Text;
    }
}
```

## 💾 Model Storage

### Automatic Caching

Models are automatically cached in the browser's Cache Storage after first download. Subsequent loads are instant.

### Export to Disk

Click **📤 Export** to save cached models to a folder on your computer. This creates:
- Model files with safe names
- `webllm-manifest.json` for re-importing

### Import from Disk

Click **📥 Import** to restore models from a previously exported folder. No re-downloading needed!

## 🐛 Troubleshooting

### "WebGPU is not supported"
- Update to Chrome 113+ or Edge 113+
- Check `chrome://gpu` for WebGPU status

### Model loading is slow
- First download can take 1-5 minutes depending on model size
- Subsequent loads use cache and are much faster

### Out of memory
- Try a smaller model (Qwen 0.5B)
- Close other browser tabs
- Restart browser to free memory

## 📚 Resources

- [WebLLM](https://webllm.mlc.ai/) - The underlying ML runtime
- [Microsoft.Extensions.AI](https://devblogs.microsoft.com/dotnet/introducing-microsoft-extensions-ai/) - AI abstractions
- [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) - GPU API for the web

## 📝 License

MIT
