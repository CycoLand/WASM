# WASM

WebAssembly experiments for running AI/LLM capabilities in the browser.

## Projects

### [experiments/blazor-webllm](experiments/blazor-webllm)

A Blazor WebAssembly application that runs LLMs entirely in the browser using [WebLLM](https://webllm.mlc.ai/). 

<img width="3840" height="2180" alt="localhost_5054_" src="https://github.com/user-attachments/assets/642d9bad-460c-46af-8426-fa68e665867c" />

**Features:**
- 🧠 In-browser LLM inference with WebGPU acceleration
- 🔧 Function calling support via cycod.core's `FunctionCallingChat`
- 💾 Model caching with import/export to local filesystem
- 📱 Fully client-side - no data sent to servers
- ⚡ Streaming responses with real-time UI updates

**Built-in Tools:**
- `GetCurrentDate` - Get the current date
- `GetCurrentTime` - Get the current time
- `GetCurrentDateTime` - Get the current date and time
- `GetDayOfWeek` - Get the current day of the week

**Supported Models:**
- Llama 3.2 (1B, 3B)
- Qwen 2.5 (0.5B, 1.5B, 3B, 7B) 
- Phi 3.5 Mini
- Gemma 2 (2B)
- SmolLM2 (135M, 360M, 1.7B)

### [external/cycod-wasm](external/cycod-wasm)

Git submodule containing the cycod.core library - provides the AI orchestration layer including:
- `FunctionCallingChat` - Agentic loop with automatic tool execution
- `FunctionFactory` - Tool registration and invocation
- `DateAndTimeHelperFunctions` - Built-in date/time tools
- Microsoft.Extensions.AI compatible abstractions

## Getting Started

### Prerequisites
- .NET 10 SDK
- A WebGPU-capable browser (Chrome 113+, Edge 113+)

### Running the Blazor WebLLM App

```bash
cd experiments/blazor-webllm
dotnet run
```

Then open https://localhost:5001 in your browser.

### First Run
1. Select a model from the dropdown (Qwen 2.5 recommended for tool support)
2. Click "Load Model" - the model will be downloaded and cached
3. Start chatting! Try asking "What time is it?" to see function calling in action

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Blazor WASM UI                       │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  ModelSelector  │  │      ChatInterface          │  │
│  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                          │                  │
│           ▼                          ▼                  │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ ModelManager    │  │    FunctionCallingChat      │  │
│  │ Service         │  │    (cycod.core)             │  │
│  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                          │                  │
│           │           ┌──────────────┴──────────────┐  │
│           │           │                             │  │
│           ▼           ▼                             ▼  │
│  ┌─────────────────────────────┐  ┌─────────────────┐  │
│  │    WebLLMChatService        │  │ FunctionFactory │  │
│  │    (IChatClient)            │  │ (Tools)         │  │
│  └──────────────┬──────────────┘  └─────────────────┘  │
│                 │                                       │
└─────────────────┼───────────────────────────────────────┘
                  │ JS Interop
                  ▼
┌─────────────────────────────────────────────────────────┐
│              WebLLM (JavaScript)                        │
│         MLCEngine + WebGPU Acceleration                 │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
