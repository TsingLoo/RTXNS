#include "SimpleInferencing.h"
#include "UserInterface.h"
#include "DeviceUtils.h"
#include "GraphicsResources.h"

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

#ifdef WIN32
#include <windows.h>
#endif

using namespace donut;

const char* g_windowTitle = "RTX Neural Shading Example: Simple Inferencing";

#ifdef WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI graphicsApi = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (graphicsApi == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("This sample does not support D3D11");
        return 1;
    }
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;
    deviceParams.depthBufferFormat = nvrhi::Format::D24S8;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableGPUValidation = false;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, true, g_windowTitle);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        }
        if (graphicsApi == nvrhi::GraphicsAPI::D3D12)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please use the Shader Model 6-9-Preview Driver, link in the README");
        }
        return 1;
    }
    auto graphicsResources = std::make_unique<rtxns::GraphicsResources>(deviceManager->GetDevice());
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        std::string gltfPath;
        for (int i = 1; i < __argc; ++i)
        {
            if (std::string(__argv[i]) == "-model" && i + 1 < __argc)
            {
                gltfPath = __argv[i + 1];
                i++;
            }
        }
        
        SimpleInferencing example(deviceManager.get(), &uiData, gltfPath);
        UserInterface gui(deviceManager.get(), &uiData, &example);

        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    return 0;
}
