using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.AI;
using Microsoft.JSInterop;

namespace BlazorWebLLM.Services;

/// <summary>
/// Implements IChatClient using WebLLM for in-browser LLM inference.
/// Compatible with Microsoft.Extensions.AI abstractions.
/// </summary>
public class WebLLMChatService : IWebLLMChatService, IAsyncDisposable
{
    private readonly IJSRuntime _jsRuntime;
    private readonly IModelManagerService _modelManager;
    private IJSObjectReference? _jsModule;
    private DotNetObjectReference<WebLLMChatService>? _dotNetRef;
    private bool _isInitialized;

    // Channel for streaming chunks from JS callbacks
    private Channel<StreamingChunk>? _streamingChannel;

    public bool IsReady => _modelManager.IsModelLoaded;
    public string? CurrentModelId => _modelManager.CurrentModelId;

    public ChatClientMetadata Metadata => new(
        providerName: "WebLLM",
        providerUri: new Uri("https://webllm.mlc.ai/"),
        defaultModelId: CurrentModelId
    );

    public WebLLMChatService(IJSRuntime jsRuntime, IModelManagerService modelManager)
    {
        _jsRuntime = jsRuntime;
        _modelManager = modelManager;
    }

    private async Task EnsureInitializedAsync()
    {
        if (_isInitialized) return;

        _jsModule = await _jsRuntime.InvokeAsync<IJSObjectReference>(
            "import", "./js/webllm-interop.js");
        _dotNetRef = DotNetObjectReference.Create(this);
        await _jsModule.InvokeVoidAsync("initializeChat", _dotNetRef);
        _isInitialized = true;
    }

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync();

        if (!IsReady)
        {
            throw new InvalidOperationException("No model is loaded. Call IModelManagerService.LoadModelAsync first.");
        }

        var messages = ConvertMessages(chatMessages.ToList());
        var requestOptions = ConvertOptions(options);

        var responseJson = await _jsModule!.InvokeAsync<string>(
            "chatComplete",
            cancellationToken,
            messages,
            requestOptions);

        return ParseChatResponse(responseJson);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync();

        if (!IsReady)
        {
            throw new InvalidOperationException("No model is loaded. Call IModelManagerService.LoadModelAsync first.");
        }

        var messages = ConvertMessages(chatMessages.ToList());
        var requestOptions = ConvertOptions(options);

        Console.WriteLine($"[WebLLM C#] Starting callback-based streaming...");

        // Create a channel for receiving chunks from JS callbacks
        _streamingChannel = Channel.CreateUnbounded<StreamingChunk>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true
        });

        // Start the streaming request - JS will call back with chunks
        // Don't await - let it run in background while we read from channel
        var streamingTask = _jsModule!.InvokeVoidAsync(
            "chatCompleteStreaming",
            cancellationToken,
            messages,
            requestOptions).AsTask();

        // Handle streaming task errors
        _ = streamingTask.ContinueWith(t =>
        {
            if (t.IsFaulted)
            {
                Console.WriteLine($"[WebLLM C#] Streaming task faulted: {t.Exception?.Message}");
                _streamingChannel?.Writer.TryComplete(t.Exception);
            }
        }, TaskScheduler.Default);

        // Read chunks from the channel as they arrive via callbacks
        await foreach (var chunk in _streamingChannel.Reader.ReadAllAsync(cancellationToken))
        {
            if (chunk.IsError)
            {
                throw new InvalidOperationException($"Streaming error: {chunk.Error}");
            }

            if (chunk.IsComplete)
            {
                // Final chunk with usage info
                if (chunk.Usage != null)
                {
                    yield return new ChatResponseUpdate
                    {
                        Role = ChatRole.Assistant,
                        Contents = [new UsageContent(new UsageDetails
                        {
                            InputTokenCount = chunk.Usage.PromptTokens,
                            OutputTokenCount = chunk.Usage.CompletionTokens,
                            TotalTokenCount = chunk.Usage.TotalTokens
                        })]
                    };
                }
                yield break;
            }

            if (!string.IsNullOrEmpty(chunk.Content))
            {
                yield return new ChatResponseUpdate
                {
                    Role = ChatRole.Assistant,
                    Contents = [new TextContent(chunk.Content)]
                };
            }
        }
    }

    /// <summary>
    /// Called from JavaScript when a streaming chunk is received.
    /// </summary>
    [JSInvokable]
    public void OnStreamingChunk(string content)
    {
        Console.WriteLine($"[WebLLM C#] OnStreamingChunk: '{content.Substring(0, Math.Min(content.Length, 20))}'");
        _streamingChannel?.Writer.TryWrite(new StreamingChunk { Content = content });
    }

    /// <summary>
    /// Called from JavaScript when streaming is complete.
    /// </summary>
    [JSInvokable]
    public void OnStreamingComplete(string? usageJson)
    {
        Console.WriteLine($"[WebLLM C#] OnStreamingComplete");
        WebLLMUsage? usage = null;
        if (!string.IsNullOrEmpty(usageJson) && usageJson != "null")
        {
            usage = JsonSerializer.Deserialize<WebLLMUsage>(usageJson, JsonOptions);
        }

        _streamingChannel?.Writer.TryWrite(new StreamingChunk { IsComplete = true, Usage = usage });
        _streamingChannel?.Writer.TryComplete();
    }

    /// <summary>
    /// Called from JavaScript when a streaming error occurs.
    /// </summary>
    [JSInvokable]
    public void OnStreamingError(string error)
    {
        Console.WriteLine($"[WebLLM C#] OnStreamingError: {error}");
        _streamingChannel?.Writer.TryWrite(new StreamingChunk { IsError = true, Error = error });
        _streamingChannel?.Writer.TryComplete();
    }

    public object? GetService(Type serviceType, object? key = null)
    {
        if (serviceType == typeof(IModelManagerService))
        {
            return _modelManager;
        }
        return null;
    }

    public void Dispose() { }

    public async ValueTask DisposeAsync()
    {
        if (_jsModule != null)
        {
            await _jsModule.DisposeAsync();
        }
        _dotNetRef?.Dispose();
    }

    #region Message and Options Conversion

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private static List<object> ConvertMessages(IList<ChatMessage> messages)
    {
        var result = new List<object>();

        foreach (var m in messages)
        {
            // Handle assistant messages that may contain tool calls
            if (m.Role == ChatRole.Assistant)
            {
                var toolCalls = m.Contents.OfType<FunctionCallContent>().ToList();
                if (toolCalls.Count > 0)
                {
                    var textContent = string.Join("", m.Contents.OfType<TextContent>().Select(c => c.Text));
                    result.Add(new
                    {
                        role = "assistant",
                        content = string.IsNullOrEmpty(textContent) ? (string?)null : textContent,
                        tool_calls = toolCalls.Select(tc => new
                        {
                            id = tc.CallId ?? Guid.NewGuid().ToString(),
                            type = "function",
                            function = new
                            {
                                name = tc.Name,
                                arguments = tc.Arguments != null
                                    ? JsonSerializer.Serialize(tc.Arguments)
                                    : "{}"
                            }
                        }).ToList()
                    });
                    continue;
                }
            }

            // Handle tool/function result messages
            if (m.Role == ChatRole.Tool)
            {
                foreach (var content in m.Contents.OfType<FunctionResultContent>())
                {
                    result.Add(new
                    {
                        role = "tool",
                        tool_call_id = content.CallId ?? "",
                        content = content.Result?.ToString() ?? ""
                    });
                }
                continue;
            }

            // Standard message (user, system, assistant with text only)
            result.Add(new
            {
                role = m.Role.Value,
                content = string.Join("", m.Contents.OfType<TextContent>().Select(c => c.Text))
            });
        }

        return result;
    }

    private static object ConvertOptions(ChatOptions? options)
    {
        var baseOptions = new Dictionary<string, object?>
        {
            ["temperature"] = options?.Temperature ?? 0.7f,
            ["max_tokens"] = options?.MaxOutputTokens ?? 512,
            ["top_p"] = options?.TopP ?? 0.9f,
            ["frequency_penalty"] = options?.FrequencyPenalty ?? 0f,
            ["presence_penalty"] = options?.PresencePenalty ?? 0f,
            ["stop"] = options?.StopSequences?.ToArray()
        };

        // Add tools if present
        var tools = options?.Tools?.OfType<AIFunction>().ToList();
        if (tools != null && tools.Count > 0)
        {
            baseOptions["tools"] = tools.Select(ConvertToolToOpenAIFormat).ToList();
            baseOptions["tool_choice"] = "auto";
        }

        return baseOptions;
    }

    private static object ConvertToolToOpenAIFormat(AIFunction function)
    {
        return new
        {
            type = "function",
            function = new
            {
                name = function.Name,
                description = function.Description ?? "",
                parameters = ConvertJsonSchema(function.JsonSchema)
            }
        };
    }

    private static object ConvertJsonSchema(JsonElement schema)
    {
        // Convert the JsonElement to a dictionary for JS interop serialization
        return JsonSerializer.Deserialize<Dictionary<string, object>>(schema.GetRawText())
            ?? new Dictionary<string, object>();
    }

    #endregion

    #region Response Parsing

    private static ChatResponse ParseChatResponse(string json)
    {
        var response = JsonSerializer.Deserialize<WebLLMResponse>(json, JsonOptions);

        if (response == null)
        {
            throw new InvalidOperationException("Failed to parse chat response");
        }

        var choice = response.Choices?.FirstOrDefault();
        var message = choice?.Message;
        var contents = new List<AIContent>();

        // Add text content if present
        if (!string.IsNullOrEmpty(message?.Content))
        {
            contents.Add(new TextContent(message.Content));
        }

        // Add tool calls if present
        if (message?.ToolCalls != null)
        {
            foreach (var toolCall in message.ToolCalls)
            {
                if (toolCall.Function != null)
                {
                    var arguments = ParseFunctionArguments(toolCall.Function.Arguments);
                    contents.Add(new FunctionCallContent(
                        toolCall.Id ?? Guid.NewGuid().ToString(),
                        toolCall.Function.Name ?? "",
                        arguments));
                }
            }
        }

        var chatMessage = new ChatMessage(ChatRole.Assistant, contents);

        return new ChatResponse([chatMessage])
        {
            ResponseId = response.Id,
            ModelId = response.Model,
            FinishReason = ParseFinishReason(choice?.FinishReason),
            Usage = response.Usage != null ? new UsageDetails
            {
                InputTokenCount = response.Usage.PromptTokens,
                OutputTokenCount = response.Usage.CompletionTokens,
                TotalTokenCount = response.Usage.TotalTokens
            } : null
        };
    }

    private static IDictionary<string, object?>? ParseFunctionArguments(string? argumentsJson)
    {
        if (string.IsNullOrEmpty(argumentsJson))
        {
            return null;
        }

        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, object?>>(argumentsJson);
        }
        catch
        {
            return null;
        }
    }

    private static ChatFinishReason? ParseFinishReason(string? reason)
    {
        return reason?.ToLowerInvariant() switch
        {
            "stop" => ChatFinishReason.Stop,
            "length" => ChatFinishReason.Length,
            "tool_calls" => ChatFinishReason.ToolCalls,
            "content_filter" => ChatFinishReason.ContentFilter,
            _ => null
        };
    }

    #endregion

    #region Internal Types

    private class StreamingChunk
    {
        public string? Content { get; init; }
        public bool IsComplete { get; init; }
        public bool IsError { get; init; }
        public string? Error { get; init; }
        public WebLLMUsage? Usage { get; init; }
    }

    private class WebLLMResponse
    {
        public string? Id { get; set; }
        public string? Model { get; set; }
        public List<WebLLMChoice>? Choices { get; set; }
        public WebLLMUsage? Usage { get; set; }
    }

    private class WebLLMChoice
    {
        public WebLLMMessage? Message { get; set; }
        public int Index { get; set; }
        public string? FinishReason { get; set; }
    }

    private class WebLLMMessage
    {
        public string? Role { get; set; }
        public string? Content { get; set; }
        public List<WebLLMToolCall>? ToolCalls { get; set; }
    }

    private class WebLLMToolCall
    {
        public string? Id { get; set; }
        public string? Type { get; set; }
        public int Index { get; set; }
        public WebLLMFunctionCall? Function { get; set; }
    }

    private class WebLLMFunctionCall
    {
        public string? Name { get; set; }
        public string? Arguments { get; set; }
    }

    private class WebLLMUsage
    {
        public int PromptTokens { get; set; }
        public int CompletionTokens { get; set; }
        public int TotalTokens { get; set; }
    }

    #endregion
}
