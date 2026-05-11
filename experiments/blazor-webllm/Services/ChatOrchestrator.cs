using Cycod.Core.Chat;
using Cycod.Core.FunctionCalling;
using Cycod.Core.Tools;
using Microsoft.Extensions.AI;

// Use explicit types to avoid ambiguity with Microsoft.Extensions.Logging
using CycodLoggerProvider = Cycod.Core.Providers.ILoggerProvider;

namespace BlazorWebLLM.Services;

/// <summary>
/// Event args for when a tool is called during chat.
/// </summary>
public class ToolCallEventArgs : EventArgs
{
    public required string ToolName { get; init; }
    public required string Arguments { get; init; }
    public required string Result { get; init; }
    public required DateTime Timestamp { get; init; }
}

/// <summary>
/// Event args for streaming text updates.
/// </summary>
public class StreamingUpdateEventArgs : EventArgs
{
    public required string Text { get; init; }
    public required bool IsComplete { get; init; }
}

/// <summary>
/// Orchestrates chat with function calling support using cycod.core's FunctionCallingChat.
/// </summary>
public class ChatOrchestrator : IDisposable
{
    private readonly IChatClient _chatClient;
    private readonly FunctionFactory _functionFactory;
    private readonly CycodLoggerProvider _logger;
    private FunctionCallingChat? _chat;
    private readonly List<ToolCallEventArgs> _toolCallHistory = [];

    private const string DefaultSystemPrompt = """
        You are a helpful AI assistant running in a web browser.
        You have access to tools that can help you answer questions accurately.
        When asked about the current date, time, or other real-time information, use the available tools.
        Always be helpful, accurate, and concise.
        """;

    public event EventHandler<ToolCallEventArgs>? ToolCalled;
    public event EventHandler<StreamingUpdateEventArgs>? StreamingUpdate;

    /// <summary>
    /// Gets the list of tool calls made during the current session.
    /// </summary>
    public IReadOnlyList<ToolCallEventArgs> ToolCallHistory => _toolCallHistory;

    /// <summary>
    /// Gets the available tools registered with the function factory.
    /// </summary>
    public IEnumerable<AITool> AvailableTools => _functionFactory.GetAITools();

    /// <summary>
    /// Gets whether the orchestrator is initialized and ready.
    /// </summary>
    public bool IsReady => _chat != null;

    public ChatOrchestrator(IChatClient chatClient, CycodLoggerProvider logger)
    {
        _chatClient = chatClient;
        _logger = logger;
        _functionFactory = new FunctionFactory(logger);

        // Register built-in WASM-safe tools
        RegisterBuiltInTools();
    }

    private void RegisterBuiltInTools()
    {
        // Add date/time tools from cycod.core
        _functionFactory.AddFunctions(new DateAndTimeHelperFunctions());

        _logger.Info($"Registered {_functionFactory.GetAITools().Count()} tools");
    }

    /// <summary>
    /// Initialize or reset the chat session.
    /// </summary>
    public void Initialize(string? systemPrompt = null)
    {
        _chat = new FunctionCallingChat(
            _chatClient,
            systemPrompt ?? DefaultSystemPrompt,
            _functionFactory,
            _logger);

        _toolCallHistory.Clear();
        _logger.Info("ChatOrchestrator initialized");
    }

    /// <summary>
    /// Send a message and get a response, with automatic tool calling.
    /// </summary>
    public async Task<string> SendMessageAsync(
        string userMessage,
        CancellationToken cancellationToken = default)
    {
        if (_chat == null)
        {
            Initialize();
        }

        var response = await _chat!.SendMessageAsync(
            userMessage,
            onStreamUpdate: content =>
            {
                StreamingUpdate?.Invoke(this, new StreamingUpdateEventArgs
                {
                    Text = content,
                    IsComplete = false
                });
            },
            onFunctionCall: (name, args, result) =>
            {
                var toolCall = new ToolCallEventArgs
                {
                    ToolName = name,
                    Arguments = args,
                    Result = result?.ToString() ?? "",
                    Timestamp = DateTime.Now
                };
                _toolCallHistory.Add(toolCall);
                ToolCalled?.Invoke(this, toolCall);
            },
            cancellationToken: cancellationToken);

        StreamingUpdate?.Invoke(this, new StreamingUpdateEventArgs
        {
            Text = response,
            IsComplete = true
        });

        return response;
    }

    /// <summary>
    /// Clear the conversation history and start fresh.
    /// </summary>
    public void ClearHistory()
    {
        _chat?.ClearChatHistory();
        _toolCallHistory.Clear();
        _logger.Info("Chat history cleared");
    }

    /// <summary>
    /// Add a custom tool to the function factory.
    /// </summary>
    public void AddTool(object toolInstance)
    {
        _functionFactory.AddFunctions(toolInstance);
        
        // Re-initialize chat to pick up new tools
        if (_chat != null)
        {
            Initialize();
        }
    }

    public void Dispose()
    {
        _chat = null;
    }
}
