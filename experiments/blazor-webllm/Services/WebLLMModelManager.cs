using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.JSInterop;

namespace BlazorWebLLM.Services;

/// <summary>
/// Implements IModelManagerService using WebLLM via JavaScript interop.
/// Manages model loading, caching, and file system operations.
/// </summary>
public class WebLLMModelManager : IModelManagerService, IAsyncDisposable
{
    private readonly IJSRuntime _jsRuntime;
    private IJSObjectReference? _jsModule;
    private DotNetObjectReference<WebLLMModelManager>? _dotNetRef;
    private bool _isInitialized;

    private static readonly List<ModelInfo> _availableModels =
    [
        // Hermes models: officially listed in WebLLM's functionCallingModelIds,
        // fine-tuned by Nous Research specifically for reliable tool calling.
        new ModelInfo
        {
            Id = "Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
            Name = "Hermes 3 - Llama 3.1 (8B)",
            Description = "Nous Research fine-tune purpose-built for tool calling - officially supported by WebLLM",
            Size = "~4.9GB",
            Quality = 5,
            Speed = 2,
            ToolSupport = ToolSupport.Excellent
        },
        new ModelInfo
        {
            Id = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
            Name = "Hermes 2 Pro - Llama 3 (8B)",
            Description = "Earlier Nous Research tool-calling fine-tune - officially supported by WebLLM",
            Size = "~5.0GB",
            Quality = 4,
            Speed = 2,
            ToolSupport = ToolSupport.Excellent
        },
        new ModelInfo
        {
            Id = "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
            Name = "Hermes 2 Pro - Mistral (7B)",
            Description = "Mistral-based tool-calling fine-tune - officially supported by WebLLM",
            Size = "~4.0GB",
            Quality = 4,
            Speed = 2,
            ToolSupport = ToolSupport.Excellent
        },
        new ModelInfo
        {
            Id = "Phi-3.5-mini-instruct-q4f16_1-MLC",
            Name = "Phi 3.5 Mini (3.8B)",
            Description = "Microsoft's powerful model - handles tools reasonably well but not in WebLLM's official tool-calling list",
            Size = "~2.1GB",
            Quality = 5,
            Speed = 2,
            ToolSupport = ToolSupport.Good
        },
        new ModelInfo
        {
            Id = "Phi-4-mini-instruct-q4f16_1-MLC",
            Name = "Phi 4 Mini",
            Description = "Successor to Phi 3.5 Mini - smaller footprint, tool support untested",
            Size = "~3.4GB",
            Quality = 5,
            Speed = 3,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "Qwen3-4B-q4f16_1-MLC",
            Name = "Qwen 3 (4B)",
            Description = "Newer Qwen generation with thinking-mode toggle - tool support untested",
            Size = "~3.4GB",
            Quality = 4,
            Speed = 3,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
            Name = "Qwen 2.5 (1.5B)",
            Description = "Good quality but not in WebLLM's official tool-calling list",
            Size = "~900MB",
            Quality = 3,
            Speed = 4,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "Qwen3-1.7B-q4f16_1-MLC",
            Name = "Qwen 3 (1.7B)",
            Description = "Small/fast Qwen3 generation - tool support untested",
            Size = "~2.0GB",
            Quality = 3,
            Speed = 4,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "Llama-3.2-1B-Instruct-q4f16_1-MLC",
            Name = "Llama 3.2 (1B)",
            Description = "Meta's small model - may work with simple tools",
            Size = "~750MB",
            Quality = 4,
            Speed = 4,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
            Name = "SmolLM2 1.7B",
            Description = "Fast and balanced - tool support untested",
            Size = "~1.0GB",
            Quality = 4,
            Speed = 4,
            ToolSupport = ToolSupport.Limited
        },
        new ModelInfo
        {
            Id = "gemma3-1b-it-q4f16_1-MLC",
            Name = "Gemma 3 (1B)",
            Description = "Tiny and fast - too small for reliable tools",
            Size = "~0.7GB",
            Quality = 3,
            Speed = 5,
            ToolSupport = ToolSupport.None
        },
        new ModelInfo
        {
            Id = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
            Name = "Qwen 2.5 (0.5B)",
            Description = "Fastest loading - too small for reliable tools",
            Size = "~350MB",
            Quality = 2,
            Speed = 5,
            ToolSupport = ToolSupport.None
        }
    ];

    public IReadOnlyList<ModelInfo> AvailableModels => _availableModels;
    public string? CurrentModelId { get; private set; }
    public bool IsModelLoaded { get; private set; }
    public bool IsLoading { get; private set; }

    public event EventHandler<ModelLoadProgress>? LoadProgressChanged;
    public event EventHandler<bool>? ModelStateChanged;
    public event EventHandler<CacheInfo>? CacheStateChanged;

    public WebLLMModelManager(IJSRuntime jsRuntime)
    {
        _jsRuntime = jsRuntime;
    }

    private async Task EnsureInitializedAsync()
    {
        if (_isInitialized) return;

        _jsModule = await _jsRuntime.InvokeAsync<IJSObjectReference>(
            "import", "./js/webllm-interop.js");
        _dotNetRef = DotNetObjectReference.Create(this);
        await _jsModule.InvokeVoidAsync("initialize", _dotNetRef);
        _isInitialized = true;
    }

    public async Task LoadModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync();

        var modelExists = _availableModels.Any(m => m.Id == modelId);
        if (!modelExists)
        {
            throw new ArgumentException($"Unknown model ID: {modelId}", nameof(modelId));
        }

        IsLoading = true;
        try
        {
            await _jsModule!.InvokeVoidAsync("loadModel", cancellationToken, modelId);
            CurrentModelId = modelId;
            IsModelLoaded = true;
            ModelStateChanged?.Invoke(this, true);
        }
        finally
        {
            IsLoading = false;
        }
    }

    public async Task UnloadModelAsync()
    {
        await EnsureInitializedAsync();
        await _jsModule!.InvokeVoidAsync("unloadModel");
        CurrentModelId = null;
        IsModelLoaded = false;
        ModelStateChanged?.Invoke(this, false);
    }

    public async Task<CacheInfo> GetCacheInfoAsync()
    {
        await EnsureInitializedAsync();
        var json = await _jsModule!.InvokeAsync<string>("getCacheInfo");
        var info = JsonSerializer.Deserialize<CacheInfo>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        });
        return info ?? new CacheInfo();
    }

    public async Task ClearCacheAsync()
    {
        await EnsureInitializedAsync();
        await _jsModule!.InvokeVoidAsync("clearCache");
        
        var cacheInfo = await GetCacheInfoAsync();
        CacheStateChanged?.Invoke(this, cacheInfo);
    }

    public async Task<StorageOperationResult> ExportModelsToFolderAsync()
    {
        await EnsureInitializedAsync();
        var json = await _jsModule!.InvokeAsync<string>("exportModelsToFolder");
        var result = JsonSerializer.Deserialize<StorageOperationResult>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        });
        return result ?? new StorageOperationResult { IsSuccess = false, ErrorMessage = "Failed to parse result" };
    }

    public async Task<StorageOperationResult> ImportModelsFromFolderAsync()
    {
        await EnsureInitializedAsync();
        var json = await _jsModule!.InvokeAsync<string>("importModelsFromFolder");
        var result = JsonSerializer.Deserialize<StorageOperationResult>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        });
        
        if (result?.IsSuccess == true)
        {
            var cacheInfo = await GetCacheInfoAsync();
            CacheStateChanged?.Invoke(this, cacheInfo);
        }
        
        return result ?? new StorageOperationResult { IsSuccess = false, ErrorMessage = "Failed to parse result" };
    }

    public async Task<bool> IsFileSystemAccessSupportedAsync()
    {
        await EnsureInitializedAsync();
        return await _jsModule!.InvokeAsync<bool>("isFileSystemAccessSupported");
    }

    /// <summary>
    /// Called from JavaScript when model loading progress changes.
    /// </summary>
    [JSInvokable]
    public void OnLoadProgress(string status, double progress, string? currentFile, long loadedBytes, long totalBytes)
    {
        var progressInfo = new ModelLoadProgress
        {
            Status = status,
            Progress = progress,
            CurrentFile = currentFile,
            LoadedBytes = loadedBytes,
            TotalBytes = totalBytes
        };
        LoadProgressChanged?.Invoke(this, progressInfo);
    }

    /// <summary>
    /// Called from JavaScript when cache state changes.
    /// </summary>
    [JSInvokable]
    public void OnCacheStateChanged(string cacheInfoJson)
    {
        var cacheInfo = JsonSerializer.Deserialize<CacheInfo>(cacheInfoJson, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        });
        if (cacheInfo != null)
        {
            CacheStateChanged?.Invoke(this, cacheInfo);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_jsModule != null)
        {
            await _jsModule.InvokeVoidAsync("dispose");
            await _jsModule.DisposeAsync();
        }
        _dotNetRef?.Dispose();
    }
}
