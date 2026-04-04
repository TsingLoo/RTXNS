#pragma once

#include <donut/app/imgui_renderer.h>
#include "UIData.h"

class SimpleInferencing;

class UserInterface : public donut::app::ImGui_Renderer
{
private:
    UIData* m_userInterfaceParameters;
    SimpleInferencing* m_app;

public:
    UserInterface(donut::app::DeviceManager* deviceManager, UIData* uiParams, SimpleInferencing* app);

    void buildUI() override;
};
