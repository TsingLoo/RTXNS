#pragma once

#include <donut/core/math/math.h>

class SimpleCamera
{
public:
    SimpleCamera();

    bool KeyboardUpdate(int key, int scancode, int action, int mods);
    bool MousePosUpdate(double xpos, double ypos);
    bool MouseButtonUpdate(int button, int action, int mods);
    bool MouseScrollUpdate(double xoffset, double yoffset);
    void Animate(float seconds);

    donut::math::float3 GetPosition() const;
    donut::math::float3 GetDir() const;
    donut::math::float3 GetUp() const;
    donut::math::float3 GetRight() const;

private:
    donut::math::float3 m_cameraTarget{ 0.f, 0.f, 0.f };
    float m_cameraAzimuth = 0.f;
    float m_cameraElevation = 0.f;
    float m_cameraDistance = 2.f;

    donut::math::float2 m_currentXY{ 0.f, 0.f };
    bool m_pressedFlag = false;
    bool m_panFlag = false;
    bool m_keys[512] = { false };
};
