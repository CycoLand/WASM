# Blazor WebLLM - Browser-Based LLM Integration Guide

> **Summary**: This document describes how to run Large Language Models (LLMs) entirely in the browser using Blazor WebAssembly, WebLLM, and WebGPU. It covers architecture, implementation details, and lessons learned for engineers integrating these capabilities into other projects.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Technologies](#key-technologies)
4. [Features](#features)
5. [Implementation Details](#implementation-details)
6. [JavaScript Interop Patterns](#javascript-interop-patterns)
7. [Streaming Implementation](#streaming-implementation)
8. [Model Caching & Storage](#model-caching--storage)
9. [Lessons Learned](#lessons-learned)
10. [Integration Guide](#integration-guide)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Overview

This project demonstrates running quantized LLMs entirely client-side in a Blazor WebAssembly application. No server is required for inference - everything runs in the browser using WebGPU for hardware acceleration.

### Why Browser-Based LLM?

| Benefit | Description |
|---------|-------------|
| **Privacy** | Data never leaves the user's device |
| **Offline capable** | Works without internet after model is cached |
| **No server costs** | Inference runs on client hardware |
| **Low latency** | No network round-trips for inference |
| **Scalability** | Each user brings their own compute |

### Trade-offs

| Limitation | Mitigation |
|------------|------------|
| Model size limited (~2-4GB) | Use quantized models (4-bit) |
| Initial download time | Browser caching + export/import to disk |
| Requires modern browser | WebGPU support check with fallback messaging |
| GPU memory constraints | Smaller models for lower-end devices |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BROWSER                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      BLAZOR WASM (.NET)                          │    │
│  │                                                                  │    │
│  │  ┌─────────────────┐     ┌─────────────────────────────────┐    │    │
│  │  │  UI Components  │     │         Services Layer          │    │    │
│  │  │                 │     │                                 │    │    │
│  │  │  ModelSelector  │     │  IModelManagerService           │    │    │
│  │  │  ChatInterface  │────►│    - LoadModelAsync()           │    │    │
│  │  │                 │     │    - ExportModelsToFolderAsync() │    │    │
│  │  │                 │     │    - ImportModelsFromFolderAsync()│    │    │
│  │  │                 │     │                                 │    │    │
│  │  │                 │     │  IWebLLMChatService : IChatClient│    │    │
│  │  │                 │────►│    - GetResponseAsync()         │    │    │
│  │  │                 │     │    - GetStreamingResponseAsync() │    │    │
│  │  └─────────────────┘     └──────────────┬──────────────────┘    │    │
│  │                                         │                        │    │
│  └─────────────────────────────────────────┼────────────────────────┘    │
│                                            │ JS Interop                  │
│  ┌─────────────────────────────────────────┼────────────────────────┐    │
│  │                      JAVASCRIPT          │                        │    │
│  │                                         ▼                        │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                 webllm-interop.js                        │    │    │
│  │  │                                                          │    │    │
│  │  │  - initialize() / initializeChat()                       │    │    │
│  │  │  - loadModel()                                           │    │    │
│  │  │  - chatComplete() / chatCompleteStreaming()              │    │    │
│  │  │  - getCacheInfo() / clearCache()                         │    │    │
│  │  │  - exportModelsToFolder() / importModelsFromFolder()     │    │    │
│  │  └──────────────────────────┬──────────────────────────────┘    │    │
│  │                             │                                    │    │
│  │                             ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                    WebLLM Engine                         │    │    │
│  │  │              (@mlc-ai/web-llm npm package)               │    │    │
│  │  │                                                          │    │    │
│  │  │  - CreateMLCEngine()                                     │    │    │
│  │  │  - engine.chat.completions.create()                      │    │    │
│  │  │  - OpenAI-compatible API                                 │    │    │
│  │  └──────────────────────────┬──────────────────────────────┘    │    │
│  │                             │                                    │    │
│  └─────────────────────────────┼────────────────────────────────────┘    │
│                                │                                         │
│  ┌─────────────────────────────┼────────────────────────────────────┐    │
│  │                             ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                       WebGPU                             │    │    │
│  │  │                                                          │    │    │
│  │  │  - GPU-accelerated tensor operations                     │    │    │
│  │  │  - Runs quantized model inference                        │    │    │
│  │  │  - Hardware: dedicated GPU, integrated GPU, or CPU       │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                                                                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      BROWSER STORAGE                              │    │
│  │                                                                   │    │
│  │  Cache API (webllm/model, webllm/wasm, etc.)                     │    │
│  │    - Model weights                                                │    │
│  │    - WASM binaries                                                │    │
│  │    - Configuration files                                          │    │
│  │                                                                   │    │
│  │  File System Access API (optional)                                │    │
│  │    - Export models to user's disk                                 │    │
│  │    - Import models from disk                                      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Streaming Chat Completion

```
User types message
        │
        ▼
┌─────────────────┐
│ ChatInterface   │
│ SendMessage()   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ WebLLMChatService               │
│ GetStreamingResponseAsync()     │
│                                 │
│ 1. Create Channel<T>            │
│ 2. Start JS call (fire & forget)│
│ 3. await foreach on Channel     │
└────────┬────────────────────────┘
         │ JS Interop
         ▼
┌─────────────────────────────────┐
│ webllm-interop.js               │
│ chatCompleteStreaming()         │
│                                 │
│ for await (chunk of stream) {   │
│   chatDotNetRef.invokeMethod    │◄──── WebLLM generates tokens
│     ("OnStreamingChunk", text)  │
│ }                               │
└────────┬────────────────────────┘
         │ Callback to C#
         ▼
┌─────────────────────────────────┐
│ WebLLMChatService               │
│ OnStreamingChunk(content)       │
│                                 │
│ channel.Writer.TryWrite(chunk)  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ ChatInterface                   │
│ await foreach (chunk in stream) │
│                                 │
│ 1. Append to StringBuilder      │
│ 2. Update _messages[index]      │
│ 3. await InvokeAsync(           │
│      StateHasChanged)           │
└─────────────────────────────────┘
         │
         ▼
    UI Re-renders
    (user sees tokens appear)
```

---

## Key Technologies

### WebLLM (@mlc-ai/web-llm)

[WebLLM](https://webllm.mlc.ai/) is a JavaScript library from MLC-AI that enables running LLMs in the browser.

**Key features:**
- OpenAI-compatible chat completions API
- Built on Apache TVM for optimized inference
- Supports many popular models (Llama, Phi, Qwen, etc.)
- Automatic WebGPU acceleration

**Example usage:**
```javascript
import * as webllm from "@mlc-ai/web-llm";

const engine = await webllm.CreateMLCEngine("Llama-3.2-1B-Instruct-q4f16_1-MLC");

// Non-streaming
const response = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Hello!" }],
    temperature: 0.7,
});

// Streaming
const stream = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Hello!" }],
    stream: true,
});
for await (const chunk of stream) {
    console.log(chunk.choices[0]?.delta?.content);
}
```

### WebGPU

WebGPU is the modern GPU API for the web, replacing WebGL for compute workloads.

**Browser support (as of 2024):**
| Browser | Support |
|---------|---------|
| Chrome 113+ | ✅ Full |
| Edge 113+ | ✅ Full |
| Firefox 141+ | ⚠️ Behind flag |
| Safari 18+ | ⚠️ Behind flag |

**Checking WebGPU availability:**
```javascript
if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No GPU adapter found");
}
```

### Microsoft.Extensions.AI

The `IChatClient` interface provides a standard abstraction for chat-based AI services.

**Key interface:**
```csharp
public interface IChatClient : IDisposable
{
    Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);

    ChatClientMetadata Metadata { get; }
    object? GetService(Type serviceType, object? key = null);
}
```

This allows our WebLLM implementation to be a drop-in replacement for other chat clients (OpenAI, Azure, etc.).

---

## Features

### 1. Model Management

**Available models:**
| Model | Parameters | Size | Quality | Speed |
|-------|------------|------|---------|-------|
| Llama 3.2 (1B) | 1B | ~750MB | ⭐⭐⭐⭐ | Fast |
| SmolLM2 1.7B | 1.7B | ~1.0GB | ⭐⭐⭐⭐ | Fast |
| Qwen 2.5 (1.5B) | 1.5B | ~900MB | ⭐⭐⭐ | Fast |
| Qwen 2.5 (0.5B) | 0.5B | ~350MB | ⭐⭐ | Fastest |
| Phi 3.5 Mini | 3.8B | ~2.1GB | ⭐⭐⭐⭐⭐ | Slower |

**Model loading with progress:**
```csharp
// Subscribe to progress events
modelManager.LoadProgressChanged += (sender, progress) =>
{
    Console.WriteLine($"{progress.Status}: {progress.Progress:F1}%");
};

// Load model
await modelManager.LoadModelAsync("Llama-3.2-1B-Instruct-q4f16_1-MLC");
```

### 2. Chat Completions

**Non-streaming:**
```csharp
var response = await chatService.GetResponseAsync(new[]
{
    new ChatMessage(ChatRole.User, "What is 3 + 9?")
});
Console.WriteLine(response.Messages.First().Text);
// Output: "3 + 9 = 12"
```

**Streaming:**
```csharp
await foreach (var update in chatService.GetStreamingResponseAsync(messages))
{
    foreach (var content in update.Contents.OfType<TextContent>())
    {
        Console.Write(content.Text); // Tokens appear incrementally
    }
}
```

### 3. Browser Caching

Models are automatically cached using the browser's Cache API:
- First load: Downloads from CDN (~1-5 minutes depending on model size)
- Subsequent loads: Instant from cache

**Cache management:**
```csharp
// Check cache status
var cacheInfo = await modelManager.GetCacheInfoAsync();
Console.WriteLine($"Cached: {cacheInfo.TotalSizeMB} MB, Files: {cacheInfo.FileCount}");

// Clear cache
await modelManager.ClearCacheAsync();
```

### 4. File System Export/Import

Using the File System Access API, users can:
- Export cached models to a folder on their disk
- Import models from disk (no re-download needed)
- Share models between browsers/computers

```csharp
// Export to user-selected folder
var result = await modelManager.ExportModelsToFolderAsync();
if (result.IsSuccess)
{
    Console.WriteLine($"Exported {result.FileCount} files to {result.FolderName}");
}

// Import from folder
var result = await modelManager.ImportModelsFromFolderAsync();
```

**Manifest file structure:**
```json
{
  "version": 1,
  "exportedAt": "2024-01-15T10:30:00Z",
  "source": "WebLLM-Blazor",
  "caches": {
    "webllm/model": [...],
    "webllm/wasm": [...]
  },
  "files": [
    {
      "cacheName": "webllm/model",
      "originalUrl": "https://...",
      "fileName": "webllm_model__Llama-3.2...",
      "size": 750000000,
      "contentType": "application/octet-stream"
    }
  ]
}
```

---

## Implementation Details

### Project Structure

```
experiments/blazor-webllm/
├── Components/
│   ├── ModelSelector.razor      # Model selection, loading, cache UI
│   └── ChatInterface.razor      # Chat UI with streaming support
├── Services/
│   ├── IWebLLMServices.cs       # Service interfaces
│   ├── Models.cs                # Data models (ModelInfo, CacheInfo, etc.)
│   ├── WebLLMModelManager.cs    # Model management implementation
│   └── WebLLMChatService.cs     # IChatClient implementation
├── wwwroot/
│   └── js/
│       └── webllm-interop.js    # JavaScript bridge to WebLLM
├── Pages/
│   └── Home.razor               # Main page
├── Program.cs                   # Service registration
└── README.md
```

### Service Registration

```csharp
// Program.cs
builder.Services.AddSingleton<IModelManagerService, WebLLMModelManager>();
builder.Services.AddSingleton<IWebLLMChatService, WebLLMChatService>();
```

**Why Singleton?**
- Both services maintain state (loaded model, JS module reference)
- State must be shared across components
- WebLLM engine should only be initialized once

---

## JavaScript Interop Patterns

### Pattern 1: Separate DotNetObjectReference per Service

We use separate .NET references for different services to keep callbacks organized:

```javascript
let dotNetRef = null;      // For ModelManager
let chatDotNetRef = null;  // For ChatService

export function initialize(dotNetReference) {
    dotNetRef = dotNetReference;
}

export function initializeChat(dotNetReference) {
    chatDotNetRef = dotNetReference;
}
```

```csharp
// In WebLLMModelManager
_dotNetRef = DotNetObjectReference.Create(this);
await _jsModule.InvokeVoidAsync("initialize", _dotNetRef);

// In WebLLMChatService
_dotNetRef = DotNetObjectReference.Create(this);
await _jsModule.InvokeVoidAsync("initializeChat", _dotNetRef);
```

### Pattern 2: Lazy Module Loading

Load the JS module only when first needed:

```csharp
private async Task EnsureInitializedAsync()
{
    if (_isInitialized) return;

    _jsModule = await _jsRuntime.InvokeAsync<IJSObjectReference>(
        "import", "./js/webllm-interop.js");
    _dotNetRef = DotNetObjectReference.Create(this);
    await _jsModule.InvokeVoidAsync("initialize", _dotNetRef);
    _isInitialized = true;
}
```

### Pattern 3: JSON Serialization for Complex Objects

Pass complex data between C# and JS using JSON:

```javascript
// JS side
return JSON.stringify({
    totalSizeBytes: totalSize,
    fileCount: keys.length,
    cachedModelIds: Array.from(cachedModelIds),
});
```

```csharp
// C# side
var json = await _jsModule.InvokeAsync<string>("getCacheInfo");
var info = JsonSerializer.Deserialize<CacheInfo>(json, new JsonSerializerOptions 
{ 
    PropertyNameCaseInsensitive = true 
});
```

---

## Streaming Implementation

### The Challenge

Blazor's JS interop doesn't naturally support streaming. We needed to:
1. Start a streaming request in JS
2. Receive tokens as they're generated
3. Update the Blazor UI in real-time

### Failed Approaches

**❌ Polling approach:**
```csharp
// This blocks and doesn't work well
while (!done) {
    var chunk = await _jsModule.InvokeAsync<string>("getNextChunk");
    yield return chunk;
}
```
Problem: JS interop calls are serialized, causing deadlocks.

**❌ Direct StateHasChanged:**
```csharp
StateHasChanged(); // Doesn't work from JS callbacks
```
Problem: Callbacks from JS aren't on the Blazor synchronization context.

### Working Solution: Channel + Callbacks

**Architecture:**
```
JS generates tokens
        │
        ▼
JS calls C# method (JSInvokable)
        │
        ▼
C# writes to Channel<T>
        │
        ▼
C# reads from Channel in async foreach
        │
        ▼
yield return update
        │
        ▼
Component calls await InvokeAsync(StateHasChanged)
```

**JavaScript side:**
```javascript
export async function chatCompleteStreaming(messages, options) {
    const stream = await engine.chat.completions.create({
        messages, stream: true, ...options
    });

    for await (const chunk of stream) {
        const content = chunk.choices?.[0]?.delta?.content || "";
        if (content && chatDotNetRef) {
            // Call back to C# for each token
            await chatDotNetRef.invokeMethodAsync("OnStreamingChunk", content);
        }
    }
    
    // Signal completion
    await chatDotNetRef.invokeMethodAsync("OnStreamingComplete", JSON.stringify(usage));
}
```

**C# service side:**
```csharp
public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(...)
{
    // Create channel for receiving chunks
    _streamingChannel = Channel.CreateUnbounded<StreamingChunkMessage>();

    // Start JS call in background (don't await - it will call back)
    _ = Task.Run(async () =>
    {
        await _jsModule.InvokeVoidAsync("chatCompleteStreaming", messages, options);
    });

    // Read from channel as chunks arrive
    await foreach (var chunk in _streamingChannel.Reader.ReadAllAsync(cancellationToken))
    {
        if (chunk.IsComplete) break;
        
        yield return new ChatResponseUpdate
        {
            Role = ChatRole.Assistant,
            Contents = [new TextContent(chunk.Content)]
        };
    }
}

[JSInvokable]
public void OnStreamingChunk(string content)
{
    _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { Content = content });
}

[JSInvokable]
public void OnStreamingComplete(string? usageJson)
{
    _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { IsComplete = true });
    _streamingChannel?.Writer.TryComplete();
}
```

**Component side:**
```csharp
await foreach (var update in ChatService.GetStreamingResponseAsync(messages))
{
    foreach (var content in update.Contents.OfType<TextContent>())
    {
        contentBuilder.Append(content.Text);
        _messages[assistantMessageIndex] = new ChatMessage(ChatRole.Assistant, contentBuilder.ToString());
        
        // CRITICAL: Must use InvokeAsync for UI updates from async streams
        await InvokeAsync(StateHasChanged);
    }
}
```

---

## Model Caching & Storage

### WebLLM Cache Structure

WebLLM uses multiple caches:
- `webllm/config` - Model configurations
- `webllm/wasm` - WebAssembly binaries
- `webllm/model` - Model weights

### Discovering Cached Content

```javascript
async function getCacheInfo() {
    const cacheNames = await caches.keys();
    
    for (const cacheName of cacheNames) {
        if (cacheName.includes("webllm") || cacheName.includes("mlc")) {
            const cache = await caches.open(cacheName);
            const keys = await cache.keys();
            // Process keys...
        }
    }
}
```

### File System Access API

**Browser support:**
- Chrome/Edge: ✅ Full support
- Firefox/Safari: ❌ Not supported

**Export flow:**
1. User clicks "Export"
2. `showDirectoryPicker()` opens folder picker
3. Iterate through all WebLLM caches
4. Write each cached response to a file
5. Create manifest JSON for re-importing

**Import flow:**
1. User clicks "Import"
2. `showDirectoryPicker()` opens folder picker
3. Read manifest.json
4. For each file in manifest:
   - Read file from disk
   - Create Response object
   - Put in appropriate cache

---

## Lessons Learned

### 1. JS Interop and Async Iterators Don't Mix Well

**Problem:** Polling JS for async iterator values from C# causes deadlocks.

**Solution:** Use callback-based approach where JS pushes to C# via `[JSInvokable]` methods.

### 2. StateHasChanged Requires Synchronization Context

**Problem:** Calling `StateHasChanged()` from JS callbacks or background tasks doesn't update UI.

**Solution:** Always use `await InvokeAsync(StateHasChanged)` when updating UI from:
- JS interop callbacks
- Background tasks
- Channel readers
- Any async context

### 3. Object Identity vs Value in Blazor

**Problem:** Updating a list item by replacing it breaks object reference tracking.

```csharp
// This breaks:
var message = new ChatMessage(...);
_messages.Add(message);
// Later:
_messages[_messages.IndexOf(message)] = newMessage; // IndexOf returns -1!
```

**Solution:** Track by index, not by reference:
```csharp
_messages.Add(new ChatMessage(...));
var index = _messages.Count - 1;
// Later:
_messages[index] = newMessage; // Always works
```

### 4. WebLLM Uses Multiple Caches

**Problem:** Looking for models in a single cache name fails.

**Solution:** Scan all caches matching patterns: `webllm`, `mlc`, `model`, `wasm`.

### 5. Channel<T> for Cross-Context Communication

**Problem:** Need to bridge JS callbacks to C# async enumerables.

**Solution:** `Channel<T>` provides a thread-safe producer-consumer pattern:
- JS callbacks write to channel
- C# async foreach reads from channel
- Clean separation of concerns

### 6. Fire-and-Forget with Error Handling

**Problem:** Need to start JS streaming without blocking, but still handle errors.

**Solution:** 
```csharp
_ = Task.Run(async () =>
{
    try
    {
        await _jsModule.InvokeVoidAsync("chatCompleteStreaming", ...);
    }
    catch (Exception ex)
    {
        _channel.Writer.TryWrite(new ErrorMessage(ex.Message));
        _channel.Writer.TryComplete();
    }
});
```

### 7. Model IDs Must Match Exactly

**Problem:** WebLLM model IDs are case-sensitive and must match exactly.

**Solution:** Use the exact IDs from WebLLM's model list:
- ✅ `Llama-3.2-1B-Instruct-q4f16_1-MLC`
- ❌ `llama-3.2-1b-instruct`

---

## Integration Guide

### Step 1: Copy Required Files

```
Your Project/
├── Services/
│   ├── IWebLLMServices.cs
│   ├── Models.cs
│   ├── WebLLMModelManager.cs
│   └── WebLLMChatService.cs
└── wwwroot/
    └── js/
        └── webllm-interop.js
```

### Step 2: Add NuGet Package

```bash
dotnet add package Microsoft.Extensions.AI.Abstractions
```

### Step 3: Register Services

```csharp
// Program.cs
builder.Services.AddSingleton<IModelManagerService, WebLLMModelManager>();
builder.Services.AddSingleton<IWebLLMChatService, WebLLMChatService>();
```

### Step 4: Use in Components

```razor
@inject IModelManagerService ModelManager
@inject IWebLLMChatService ChatService

@code {
    protected override async Task OnInitializedAsync()
    {
        // Subscribe to state changes
        ModelManager.ModelStateChanged += (s, loaded) => InvokeAsync(StateHasChanged);
        
        // Load a model
        await ModelManager.LoadModelAsync("Qwen2.5-0.5B-Instruct-q4f16_1-MLC");
    }

    private async Task Chat(string userMessage)
    {
        var messages = new[] { new ChatMessage(ChatRole.User, userMessage) };
        
        await foreach (var update in ChatService.GetStreamingResponseAsync(messages))
        {
            // Handle streaming updates...
            await InvokeAsync(StateHasChanged);
        }
    }
}
```

### Step 5: Add WebGPU Check

```razor
@if (!_webGpuSupported)
{
    <div class="alert alert-danger">
        WebGPU is not supported. Please use Chrome 113+ or Edge 113+.
    </div>
}
```

---

## Troubleshooting

### "WebGPU is not supported"

**Cause:** Browser doesn't support WebGPU or it's disabled.

**Solutions:**
- Update to Chrome 113+ or Edge 113+
- Firefox: Enable `dom.webgpu.enabled` in `about:config`
- Safari: Enable in Develop → Experimental Features → WebGPU

### Model loading hangs

**Cause:** Network issues or CORS problems.

**Solutions:**
- Check browser console for network errors
- Ensure CDN URLs are accessible
- Try a smaller model first

### Streaming shows only first token

**Cause:** `StateHasChanged()` not being called correctly.

**Solution:** Use `await InvokeAsync(StateHasChanged)` instead of `StateHasChanged()`.

### Export says "No models cached"

**Cause:** Looking in wrong cache names.

**Solution:** Scan all caches matching `webllm`, `mlc`, `model`, `wasm` patterns.

### Chat input stays disabled after model loads

**Cause:** Component not re-rendering when model state changes.

**Solution:** Subscribe to `ModelStateChanged` event and call `InvokeAsync(StateHasChanged)`.

---

## References

### Documentation
- [WebLLM Documentation](https://webllm.mlc.ai/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [Microsoft.Extensions.AI](https://devblogs.microsoft.com/dotnet/introducing-microsoft-extensions-ai/)
- [Blazor JS Interop](https://learn.microsoft.com/en-us/aspnet/core/blazor/javascript-interoperability/)

### Model Sources
- [MLC-AI Model Hub](https://huggingface.co/mlc-ai)
- [WebLLM Model List](https://webllm.mlc.ai/#702702)

### Related Projects
- [Transformers.js](https://huggingface.co/docs/transformers.js) - Alternative using ONNX Runtime
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) - Lower-level ONNX inference

---

## Summary

This project demonstrates that running LLMs entirely in the browser is not only possible but practical for many use cases. The key architectural decisions were:

1. **WebLLM for inference** - Provides optimized WebGPU-accelerated inference with an OpenAI-compatible API
2. **Channel<T> for streaming** - Bridges JS callbacks to C# async enumerables
3. **Callback-based JS interop** - Avoids deadlocks with async iterators
4. **IChatClient compatibility** - Allows drop-in replacement with other AI providers
5. **File System Access API** - Enables model portability and backup

The result is a fully client-side AI chat application that protects user privacy, works offline, and requires no server infrastructure for inference.
