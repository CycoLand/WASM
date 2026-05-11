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
    private Channel<StreamingChunkMessage>? _streamingChannel;
    
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

        var messages = ConvertMessages(chatMessages);
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

        var messages = ConvertMessages(chatMessages);
        var requestOptions = ConvertOptions(options);

        // Create a channel for receiving streaming chunks
        _streamingChannel = Channel.CreateUnbounded<StreamingChunkMessage>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false
        });

        // Start the streaming request in the background (don't await - it will call back)
        _ = Task.Run(async () =>
        {
            try
            {
                await _jsModule!.InvokeVoidAsync(
                    "chatCompleteStreaming",
                    cancellationToken,
                    messages,
                    requestOptions);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WebLLM C#] Streaming task error: {ex.Message}");
                _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { IsError = true, Error = ex.Message });
                _streamingChannel?.Writer.TryComplete();
            }
        });

        // Read chunks from the channel as they arrive
        await foreach (var chunk in _streamingChannel.Reader.ReadAllAsync(cancellationToken))
        {
            Console.WriteLine($"[WebLLM C#] Read chunk from channel: Content='{chunk.Content}', IsComplete={chunk.IsComplete}, IsError={chunk.IsError}");
            
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
                break;
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
        Console.WriteLine($"[WebLLM C#] OnStreamingChunk called: '{content}'");
        var written = _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { Content = content });
        Console.WriteLine($"[WebLLM C#] Written to channel: {written}");
    }

    /// <summary>
    /// Called from JavaScript when streaming is complete.
    /// </summary>
    [JSInvokable]
    public void OnStreamingComplete(string? usageJson)
    {
        Console.WriteLine($"[WebLLM C#] OnStreamingComplete called");
        WebLLMUsage? usage = null;
        if (!string.IsNullOrEmpty(usageJson) && usageJson != "null")
        {
            usage = JsonSerializer.Deserialize<WebLLMUsage>(usageJson, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
        }
        
        _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { IsComplete = true, Usage = usage });
        _streamingChannel?.Writer.TryComplete();
    }

    /// <summary>
    /// Called from JavaScript when a streaming error occurs.
    /// </summary>
    [JSInvokable]
    public void OnStreamingError(string error)
    {
        _streamingChannel?.Writer.TryWrite(new StreamingChunkMessage { IsError = true, Error = error });
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

    private static List<object> ConvertMessages(IEnumerable<ChatMessage> messages)
    {
        return messages.Select(m => new
        {
            role = m.Role.Value,
            content = string.Join("", m.Contents.OfType<TextContent>().Select(c => c.Text))
        }).ToList<object>();
    }

    private static object ConvertOptions(ChatOptions? options)
    {
        return new
        {
            temperature = options?.Temperature ?? 0.7f,
            max_tokens = options?.MaxOutputTokens ?? 512,
            top_p = options?.TopP ?? 0.9f,
            frequency_penalty = options?.FrequencyPenalty ?? 0f,
            presence_penalty = options?.PresencePenalty ?? 0f,
            stop = options?.StopSequences?.ToArray()
        };
    }

    private static ChatResponse ParseChatResponse(string json)
    {
        var response = JsonSerializer.Deserialize<WebLLMResponse>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        if (response == null)
        {
            throw new InvalidOperationException("Failed to parse chat response");
        }

        var content = response.Choices?.FirstOrDefault()?.Message?.Content ?? "";

        return new ChatResponse([new ChatMessage(ChatRole.Assistant, content)])
        {
            ResponseId = response.Id,
            ModelId = response.Model,
            Usage = response.Usage != null ? new UsageDetails
            {
                InputTokenCount = response.Usage.PromptTokens,
                OutputTokenCount = response.Usage.CompletionTokens,
                TotalTokenCount = response.Usage.TotalTokens
            } : null
        };
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

    // Internal message type for the streaming channel
    private class StreamingChunkMessage
    {
        public string? Content { get; init; }
        public bool IsComplete { get; init; }
        public bool IsError { get; init; }
        public string? Error { get; init; }
        public WebLLMUsage? Usage { get; init; }
    }

    // Internal classes for JSON deserialization
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
    }

    private class WebLLMUsage
    {
        public int PromptTokens { get; set; }
        public int CompletionTokens { get; set; }
        public int TotalTokens { get; set; }
    }
}
