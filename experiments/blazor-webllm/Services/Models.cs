namespace BlazorWebLLM.Services;

/// <summary>
/// Indicates the level of function/tool calling support for a model.
/// </summary>
public enum ToolSupport
{
    /// <summary>Model does not support function calling.</summary>
    None,
    /// <summary>Model may work with simple tools but is not reliable.</summary>
    Limited,
    /// <summary>Model supports function calling well.</summary>
    Good,
    /// <summary>Model is specifically optimized for function calling.</summary>
    Excellent
}

/// <summary>
/// Represents information about an available LLM model.
/// </summary>
public class ModelInfo
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required string Description { get; init; }
    public required string Size { get; init; }
    public required int Quality { get; init; }
    public required int Speed { get; init; }
    public ToolSupport ToolSupport { get; init; } = ToolSupport.None;
    public bool IsCached { get; set; }
}

/// <summary>
/// Progress information for model loading operations.
/// </summary>
public class ModelLoadProgress
{
    public required string Status { get; init; }
    public double Progress { get; init; }
    public string? CurrentFile { get; init; }
    public long LoadedBytes { get; init; }
    public long TotalBytes { get; init; }
}

/// <summary>
/// Information about the current cache state.
/// </summary>
public class CacheInfo
{
    public long TotalSizeBytes { get; init; }
    public string TotalSizeMB => (TotalSizeBytes / 1024.0 / 1024.0).ToString("F1");
    public int FileCount { get; init; }
    public IReadOnlyList<string> CachedModelIds { get; init; } = [];
}

/// <summary>
/// Result of an export or import operation.
/// </summary>
public class StorageOperationResult
{
    public bool IsSuccess { get; init; }
    public string? ErrorMessage { get; init; }
    public int FileCount { get; init; }
    public long TotalBytes { get; init; }
    public string? FolderName { get; init; }
}
