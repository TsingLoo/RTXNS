#include "UserInterface.h"
#include "SimpleInferencing.h"
#include <donut/app/UserInterfaceUtils.h>

using namespace donut;

UserInterface::UserInterface(app::DeviceManager* deviceManager, UIData* uiParams, SimpleInferencing* app)
    : ImGui_Renderer(deviceManager), m_userInterfaceParameters(uiParams), m_app(app)
{
    ImGui::GetIO().IniFilename = nullptr;
}

void UserInterface::buildUI()
{
    ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::Button("Load GLTF Model"))
    {
        std::string fileName;
        if (donut::app::FileDialog(true, "GLTF Files\0*.gltf;*.glb\0All Files\0*.*\0\0", fileName))
        {
            m_app->LoadModel(fileName);
        }
    }
    
    if (ImGui::Button("Load HDRI Skybox"))
    {
        std::string fileName;
        if (donut::app::FileDialog(true, "HDR Files\0*.hdr;*.exr\0All Files\0*.*\0\0", fileName))
        {
            m_app->LoadSkyboxTexture(fileName);
        }
    }
    ImGui::Separator();

    ImGui::SliderFloat3("Light Direction", &m_userInterfaceParameters->lightDir.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Intensity", &m_userInterfaceParameters->lightIntensity, 0.f, 20.f);
    ImGui::SliderFloat("Specular", &m_userInterfaceParameters->specular, 0.f, 1.f);
    ImGui::SliderFloat("Roughness", &m_userInterfaceParameters->roughness, 0.3f, 1.f);
    ImGui::SliderFloat("Metallic", &m_userInterfaceParameters->metallic, 0.f, 1.f);
    ImGui::Checkbox("Enable Neural Shading", &m_userInterfaceParameters->enableNeuralShading);

    ImGui::End();
}
