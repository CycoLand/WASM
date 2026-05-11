namespace BlazorWebLLM.Services;

// Use explicit types to avoid ambiguity with Microsoft.Extensions.Logging
using CycodLoggerProvider = Cycod.Core.Providers.ILoggerProvider;
using CycodLogLevel = Cycod.Core.Providers.LogLevel;

/// <summary>
/// Simple logger implementation for Blazor WASM that logs to the browser console.
/// </summary>
public class BrowserLoggerProvider : CycodLoggerProvider
{
    private readonly List<string> _memoryLogs = [];
    private readonly CycodLogLevel _minLevel;

    public BrowserLoggerProvider(CycodLogLevel minLevel = CycodLogLevel.Info)
    {
        _minLevel = minLevel;
    }

    public void Log(CycodLogLevel level, string message, string? filePath = null, int lineNumber = 0)
    {
        if (!IsLevelEnabled(level)) return;

        var prefix = level switch
        {
            CycodLogLevel.Error => "[ERROR]",
            CycodLogLevel.Warning => "[WARN]",
            CycodLogLevel.Info => "[INFO]",
            CycodLogLevel.Verbose => "[VERBOSE]",
            _ => "[LOG]"
        };

        var logMessage = $"{prefix} {message}";
        _memoryLogs.Add(logMessage);

        // Keep memory logs bounded
        if (_memoryLogs.Count > 1000)
        {
            _memoryLogs.RemoveAt(0);
        }

        Console.WriteLine(logMessage);
    }

    public void Error(string message) => Log(CycodLogLevel.Error, message);
    public void Warning(string message) => Log(CycodLogLevel.Warning, message);
    public void Info(string message) => Log(CycodLogLevel.Info, message);
    public void Verbose(string message) => Log(CycodLogLevel.Verbose, message);

    public bool IsLevelEnabled(CycodLogLevel level) => level <= _minLevel;

    public IEnumerable<string>? GetMemoryLogs(int? maxLines = null)
    {
        if (maxLines.HasValue)
        {
            return _memoryLogs.TakeLast(maxLines.Value);
        }
        return _memoryLogs;
    }
}
