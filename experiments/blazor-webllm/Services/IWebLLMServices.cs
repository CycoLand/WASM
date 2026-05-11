using Microsoft.Extensions.AI;

namespace BlazorWebLLM.Services;

/// <summary>
/// Service interface for managing LLM models in the browser.
/// Handles model loading, caching, and file system operations.
/// </summary>
public interface IModelManagerService
{
    /// <summary>
    /// Gets the list of available models that can be loaded.
    /// </summary>
    IReadOnlyList<ModelInfo> AvailableModels { get; }

    /// <summary>
    /// Gets the currently loaded model ID, or null if no model is loaded.
    /// </summary>
    string? CurrentModelId { get; }

    /// <summary>
    /// Gets whether a model is currently loaded and ready for inference.
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    /// Gets whether a model is currently being loaded.
    /// </summary>
    bool IsLoading { get; }

    /// <summary>
    /// Event raised when model loading progress changes.
    /// </summary>
    event EventHandler<ModelLoadProgress>? LoadProgressChanged;

    /// <summary>
    /// Event raised when the model state changes (loaded/unloaded).
    /// </summary>
    event EventHandler<bool>? ModelStateChanged;

    /// <summary>
    /// Event raised when the cache state changes.
    /// </summary>
    event EventHandler<CacheInfo>? CacheStateChanged;

    /// <summary>
    /// Loads a model by its ID. Progress is reported via LoadProgressChanged event.
    /// </summary>
    Task LoadModelAsync(string modelId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Unloads the current model and frees resources.
    /// </summary>
    Task UnloadModelAsync();

    /// <summary>
    /// Gets information about the current cache state.
    /// </summary>
    Task<CacheInfo> GetCacheInfoAsync();

    /// <summary>
    /// Clears all cached models from browser storage.
    /// </summary>
    Task ClearCacheAsync();

    /// <summary>
    /// Exports cached models to a user-selected folder using File System Access API.
    /// </summary>
    Task<StorageOperationResult> ExportModelsToFolderAsync();

    /// <summary>
    /// Imports models from a user-selected folder using File System Access API.
    /// </summary>
    Task<StorageOperationResult> ImportModelsFromFolderAsync();

    /// <summary>
    /// Checks if File System Access API is supported in the current browser.
    /// </summary>
    Task<bool> IsFileSystemAccessSupportedAsync();
}

/// <summary>
/// Service interface for LLM chat completions.
/// Implements Microsoft.Extensions.AI.IChatClient for compatibility.
/// </summary>
public interface IWebLLMChatService : IChatClient
{
    /// <summary>
    /// Gets whether the service is ready to process requests.
    /// </summary>
    bool IsReady { get; }

    /// <summary>
    /// Gets the ID of the currently loaded model.
    /// </summary>
    string? CurrentModelId { get; }
}
