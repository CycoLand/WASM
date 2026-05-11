using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using BlazorWebLLM;
using BlazorWebLLM.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// Register WebLLM services
builder.Services.AddSingleton<IModelManagerService, WebLLMModelManager>();
builder.Services.AddSingleton<IWebLLMChatService, WebLLMChatService>();

await builder.Build().RunAsync();
