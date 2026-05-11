/**
 * WebLLM JavaScript Interop for Blazor
 * 
 * This module provides the bridge between Blazor C# code and WebLLM.
 * It handles model loading, chat completions, caching, and file system operations.
 */

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// ============================================================================
// State
// ============================================================================

let engine = null;
let dotNetRef = null;
let currentModelId = null;
let activeStreams = new Map();
let streamIdCounter = 0;

// ============================================================================
// Initialization
// ============================================================================

let chatDotNetRef = null;

export function initialize(dotNetReference) {
    dotNetRef = dotNetReference;
    console.log("[WebLLM] Initialized with .NET reference (ModelManager)");
}

export function initializeChat(dotNetReference) {
    chatDotNetRef = dotNetReference;
    console.log("[WebLLM] Initialized with .NET reference (ChatService)");
}

export function dispose() {
    if (engine) {
        engine.unload();
        engine = null;
    }
    activeStreams.clear();
    dotNetRef = null;
    chatDotNetRef = null;
    console.log("[WebLLM] Disposed");
}

// ============================================================================
// Model Management
// ============================================================================

export async function loadModel(modelId) {
    console.log(`[WebLLM] Loading model: ${modelId}`);
    
    // Unload existing model if any
    if (engine) {
        await engine.unload();
        engine = null;
    }

    // Create progress callback
    const progressCallback = (progress) => {
        if (dotNetRef) {
            const status = progress.text || "Loading...";
            const progressPercent = progress.progress || 0;
            console.log(`[WebLLM] Progress: ${status} (${(progressPercent * 100).toFixed(1)}%)`);
            dotNetRef.invokeMethodAsync(
                "OnLoadProgress",
                status,
                progressPercent * 100,
                null,
                0,
                0
            );
        }
    };

    try {
        // Initialize the engine with the selected model
        engine = await webllm.CreateMLCEngine(modelId, {
            initProgressCallback: progressCallback,
        });

        currentModelId = modelId;
        console.log(`[WebLLM] Model loaded successfully: ${modelId}`);
        
        // Notify final progress
        if (dotNetRef) {
            dotNetRef.invokeMethodAsync("OnLoadProgress", "Ready", 100, null, 0, 0);
        }
        
        return JSON.stringify({ success: true, modelId: modelId });
    } catch (error) {
        console.error(`[WebLLM] Failed to load model: ${error.message}`);
        engine = null;
        currentModelId = null;
        throw error;
    }
}

export function isModelLoaded() {
    return engine !== null && currentModelId !== null;
}

export function getCurrentModelId() {
    return currentModelId;
}

export async function unloadModel() {
    if (engine) {
        await engine.unload();
        engine = null;
        currentModelId = null;
        console.log("[WebLLM] Model unloaded");
    }
}

// ============================================================================
// Chat Completions (Non-Streaming)
// ============================================================================

export async function chatComplete(messages, options) {
    if (!engine) {
        throw new Error("No model loaded");
    }

    const response = await engine.chat.completions.create({
        messages: messages,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        stop: options.stop,
        stream: false,
    });

    return JSON.stringify(response);
}

// ============================================================================
// Chat Completions (Streaming)
// ============================================================================

// Callback-based streaming - JS calls .NET for each chunk
export async function chatCompleteStreaming(messages, options) {
    if (!engine) {
        throw new Error("No model loaded");
    }

    console.log("[WebLLM] Starting streaming chat completion...");
    
    const chunks = [];
    let usage = null;
    
    try {
        const stream = await engine.chat.completions.create({
            messages: messages,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            top_p: options.top_p,
            frequency_penalty: options.frequency_penalty,
            presence_penalty: options.presence_penalty,
            stop: options.stop,
            stream: true,
            stream_options: { include_usage: true },
        });

        for await (const chunk of stream) {
            const content = chunk.choices?.[0]?.delta?.content || "";
            if (content) {
                chunks.push(content);
                console.log(`[WebLLM] Chunk: "${content}"`);
                
                // Call back to .NET with each chunk
                if (chatDotNetRef) {
                    await chatDotNetRef.invokeMethodAsync("OnStreamingChunk", content);
                }
            }
            
            // Capture usage from final chunk
            if (chunk.usage) {
                usage = {
                    promptTokens: chunk.usage.prompt_tokens,
                    completionTokens: chunk.usage.completion_tokens,
                    totalTokens: chunk.usage.total_tokens,
                };
            }
        }
        
        console.log(`[WebLLM] Streaming complete. Total chunks: ${chunks.length}`);
        
        // Notify completion
        if (chatDotNetRef) {
            await chatDotNetRef.invokeMethodAsync("OnStreamingComplete", JSON.stringify(usage));
        }
        
        return JSON.stringify({
            success: true,
            content: chunks.join(""),
            usage: usage
        });
    } catch (error) {
        console.error("[WebLLM] Streaming error:", error);
        if (chatDotNetRef) {
            await chatDotNetRef.invokeMethodAsync("OnStreamingError", error.message);
        }
        throw error;
    }
}

// Legacy polling-based streaming (keeping for reference)
export async function chatCompleteStreamingStart(messages, options) {
    if (!engine) {
        throw new Error("No model loaded");
    }

    const streamId = `stream_${++streamIdCounter}`;
    
    const stream = await engine.chat.completions.create({
        messages: messages,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        stop: options.stop,
        stream: true,
        stream_options: { include_usage: true },
    });

    // Store the async iterator
    activeStreams.set(streamId, {
        iterator: stream[Symbol.asyncIterator](),
        done: false,
        usage: null,
    });

    return streamId;
}

export async function chatCompleteStreamingNext(streamId) {
    const streamState = activeStreams.get(streamId);
    if (!streamState) {
        console.log(`[WebLLM] Stream ${streamId} not found`);
        return JSON.stringify({ isDone: true });
    }

    if (streamState.done) {
        console.log(`[WebLLM] Stream ${streamId} already done`);
        activeStreams.delete(streamId);
        return JSON.stringify({ isDone: true, usage: streamState.usage });
    }

    try {
        console.log(`[WebLLM] Getting next chunk for stream ${streamId}...`);
        const { value, done } = await streamState.iterator.next();
        
        if (done) {
            console.log(`[WebLLM] Stream ${streamId} iterator done`);
            streamState.done = true;
            activeStreams.delete(streamId);
            return JSON.stringify({ isDone: true, usage: streamState.usage });
        }

        // Extract content from the chunk
        const content = value.choices?.[0]?.delta?.content || "";
        console.log(`[WebLLM] Stream ${streamId} chunk: "${content}"`);
        
        // Check for usage info (sent in final chunk)
        if (value.usage) {
            streamState.usage = {
                promptTokens: value.usage.prompt_tokens,
                completionTokens: value.usage.completion_tokens,
                totalTokens: value.usage.total_tokens,
            };
        }

        // Check if this is the final content chunk
        const finishReason = value.choices?.[0]?.finish_reason;
        if (finishReason) {
            streamState.done = true;
        }

        return JSON.stringify({
            content: content,
            isDone: false,
        });
    } catch (error) {
        console.error("[WebLLM] Streaming error:", error);
        activeStreams.delete(streamId);
        return JSON.stringify({ isDone: true, error: error.message });
    }
}

// ============================================================================
// Cache Management
// ============================================================================

// WebLLM uses multiple cache names - we need to check all of them
const WEBLLM_CACHE_NAMES = [
    "webllm/config",
    "webllm/wasm", 
    "webllm/model",
    "transformers-cache",
    "webllm-cache"
];

async function getAllWebLLMCaches() {
    const allKeys = [];
    const cacheNames = await caches.keys();
    
    for (const cacheName of cacheNames) {
        // Include any cache that might be WebLLM related
        if (cacheName.includes("webllm") || 
            cacheName.includes("mlc") || 
            cacheName.includes("model") ||
            cacheName.includes("wasm")) {
            try {
                const cache = await caches.open(cacheName);
                const keys = await cache.keys();
                for (const key of keys) {
                    allKeys.push({ cacheName, request: key });
                }
            } catch (e) {
                console.warn(`[WebLLM] Could not open cache ${cacheName}:`, e);
            }
        }
    }
    return allKeys;
}

export async function getCacheInfo() {
    try {
        let totalSize = 0;
        const cachedModelIds = new Set();
        let fileCount = 0;
        
        // Get all cache names
        const cacheNames = await caches.keys();
        console.log("[WebLLM] Available caches:", cacheNames);
        
        for (const cacheName of cacheNames) {
            // Check caches that might contain WebLLM data
            if (cacheName.includes("webllm") || 
                cacheName.includes("mlc") || 
                cacheName.includes("model") ||
                cacheName.includes("wasm") ||
                cacheName.includes("huggingface")) {
                try {
                    const cache = await caches.open(cacheName);
                    const keys = await cache.keys();
                    
                    for (const request of keys) {
                        const response = await cache.match(request);
                        if (response) {
                            const blob = await response.clone().blob();
                            totalSize += blob.size;
                            fileCount++;
                            
                            // Try to extract model ID from URL
                            const url = request.url;
                            const modelMatch = url.match(/([A-Za-z0-9._-]+-MLC)/i);
                            if (modelMatch) {
                                cachedModelIds.add(modelMatch[1]);
                            }
                        }
                    }
                } catch (e) {
                    console.warn(`[WebLLM] Could not read cache ${cacheName}:`, e);
                }
            }
        }
        
        console.log(`[WebLLM] Cache info: ${fileCount} files, ${(totalSize/1024/1024).toFixed(1)} MB, models: ${Array.from(cachedModelIds).join(', ')}`);
        
        return JSON.stringify({
            totalSizeBytes: totalSize,
            fileCount: fileCount,
            cachedModelIds: Array.from(cachedModelIds),
        });
    } catch (error) {
        console.error("[WebLLM] Error getting cache info:", error);
        return JSON.stringify({
            totalSizeBytes: 0,
            fileCount: 0,
            cachedModelIds: [],
        });
    }
}

export async function clearCache() {
    try {
        // Get all cache names and delete WebLLM-related ones
        const cacheNames = await caches.keys();
        for (const cacheName of cacheNames) {
            if (cacheName.includes("webllm") || 
                cacheName.includes("mlc") || 
                cacheName.includes("model") ||
                cacheName.includes("wasm")) {
                await caches.delete(cacheName);
                console.log(`[WebLLM] Deleted cache: ${cacheName}`);
            }
        }
        
        // Also try to clear any IndexedDB data
        const databases = await indexedDB.databases?.() || [];
        for (const db of databases) {
            if (db.name && (db.name.includes("webllm") || db.name.includes("mlc"))) {
                indexedDB.deleteDatabase(db.name);
                console.log(`[WebLLM] Deleted IndexedDB: ${db.name}`);
            }
        }
        
        console.log("[WebLLM] Cache cleared");
        return true;
    } catch (error) {
        console.error("[WebLLM] Error clearing cache:", error);
        return false;
    }
}

// ============================================================================
// File System Access API
// ============================================================================

export function isFileSystemAccessSupported() {
    return "showDirectoryPicker" in window;
}

export async function exportModelsToFolder() {
    if (!isFileSystemAccessSupported()) {
        return JSON.stringify({
            isSuccess: false,
            errorMessage: "File System Access API is not supported in this browser.",
        });
    }

    try {
        const dirHandle = await window.showDirectoryPicker({
            mode: "readwrite",
            startIn: "downloads",
        });

        const manifest = {
            version: 1,
            exportedAt: new Date().toISOString(),
            source: "WebLLM-Blazor",
            caches: {},
            files: [],
        };

        let exportedCount = 0;
        let totalBytes = 0;

        // Get all cache names
        const cacheNames = await caches.keys();
        
        for (const cacheName of cacheNames) {
            // Check caches that might contain WebLLM data
            if (cacheName.includes("webllm") || 
                cacheName.includes("mlc") || 
                cacheName.includes("model") ||
                cacheName.includes("wasm") ||
                cacheName.includes("huggingface")) {
                
                const cache = await caches.open(cacheName);
                const keys = await cache.keys();
                
                if (keys.length === 0) continue;
                
                manifest.caches[cacheName] = [];
                
                for (const request of keys) {
                    const response = await cache.match(request);
                    if (!response) continue;

                    const blob = await response.blob();
                    const url = new URL(request.url);
                    // Create safe filename with cache name prefix
                    const safeName = `${cacheName.replace(/\//g, "_")}__${url.pathname.replace(/^\//, "").replace(/\//g, "__")}`;

                    try {
                        const fileHandle = await dirHandle.getFileHandle(safeName, { create: true });
                        const writable = await fileHandle.createWritable();
                        await writable.write(blob);
                        await writable.close();

                        const fileInfo = {
                            cacheName: cacheName,
                            originalUrl: request.url,
                            fileName: safeName,
                            size: blob.size,
                            contentType: response.headers.get("content-type") || "application/octet-stream",
                        };
                        
                        manifest.files.push(fileInfo);
                        manifest.caches[cacheName].push(fileInfo);

                        exportedCount++;
                        totalBytes += blob.size;
                        
                        console.log(`[WebLLM] Exported: ${safeName} (${(blob.size/1024/1024).toFixed(2)} MB)`);
                    } catch (fileError) {
                        console.warn(`[WebLLM] Failed to export ${safeName}:`, fileError);
                    }
                }
            }
        }

        if (exportedCount === 0) {
            return JSON.stringify({
                isSuccess: false,
                errorMessage: "No models cached to export. Load a model first.",
            });
        }

        // Save manifest
        const manifestHandle = await dirHandle.getFileHandle("webllm-manifest.json", { create: true });
        const manifestWritable = await manifestHandle.createWritable();
        await manifestWritable.write(JSON.stringify(manifest, null, 2));
        await manifestWritable.close();

        console.log(`[WebLLM] Export complete: ${exportedCount} files, ${(totalBytes/1024/1024).toFixed(1)} MB`);

        return JSON.stringify({
            isSuccess: true,
            fileCount: exportedCount,
            totalBytes: totalBytes,
            folderName: dirHandle.name,
        });
    } catch (error) {
        if (error.name === "AbortError") {
            return JSON.stringify({
                isSuccess: false,
                errorMessage: "Export cancelled by user.",
            });
        }
        console.error("[WebLLM] Export error:", error);
        return JSON.stringify({
            isSuccess: false,
            errorMessage: error.message,
        });
    }
}

export async function importModelsFromFolder() {
    if (!isFileSystemAccessSupported()) {
        return JSON.stringify({
            isSuccess: false,
            errorMessage: "File System Access API is not supported in this browser.",
        });
    }

    try {
        const dirHandle = await window.showDirectoryPicker({
            mode: "read",
            startIn: "downloads",
        });

        // Read manifest
        let manifest;
        try {
            const manifestHandle = await dirHandle.getFileHandle("webllm-manifest.json");
            const manifestFile = await manifestHandle.getFile();
            manifest = JSON.parse(await manifestFile.text());
        } catch (e) {
            return JSON.stringify({
                isSuccess: false,
                errorMessage: "Could not find webllm-manifest.json in the selected folder.",
            });
        }

        let importedCount = 0;
        let totalBytes = 0;

        // Group files by cache name
        const cacheGroups = {};
        for (const fileInfo of manifest.files) {
            const cacheName = fileInfo.cacheName || "webllm/model";
            if (!cacheGroups[cacheName]) {
                cacheGroups[cacheName] = [];
            }
            cacheGroups[cacheName].push(fileInfo);
        }

        // Import to each cache
        for (const [cacheName, files] of Object.entries(cacheGroups)) {
            const cache = await caches.open(cacheName);
            
            for (const fileInfo of files) {
                try {
                    const fileHandle = await dirHandle.getFileHandle(fileInfo.fileName);
                    const file = await fileHandle.getFile();

                    const response = new Response(file, {
                        headers: {
                            "content-type": fileInfo.contentType,
                            "content-length": fileInfo.size.toString(),
                        },
                    });

                    await cache.put(fileInfo.originalUrl, response);
                    importedCount++;
                    totalBytes += fileInfo.size;
                    
                    console.log(`[WebLLM] Imported: ${fileInfo.fileName} to ${cacheName}`);
                } catch (fileError) {
                    console.warn(`[WebLLM] Failed to import ${fileInfo.fileName}:`, fileError);
                }
            }
        }

        console.log(`[WebLLM] Import complete: ${importedCount} files, ${(totalBytes/1024/1024).toFixed(1)} MB`);

        return JSON.stringify({
            isSuccess: true,
            fileCount: importedCount,
            totalBytes: totalBytes,
            folderName: dirHandle.name,
        });
    } catch (error) {
        if (error.name === "AbortError") {
            return JSON.stringify({
                isSuccess: false,
                errorMessage: "Import cancelled by user.",
            });
        }
        console.error("[WebLLM] Import error:", error);
        return JSON.stringify({
            isSuccess: false,
            errorMessage: error.message,
        });
    }
}
