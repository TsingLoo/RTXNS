#include "SimpleCamera.h"
#include <GLFW/glfw3.h>
#include <algorithm>

using namespace donut::math;

SimpleCamera::SimpleCamera()
{
    cartesianToSpherical(float3(0.f, 0.f, 2.f), m_cameraAzimuth, m_cameraElevation, m_cameraDistance);
}

bool SimpleCamera::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (key >= 0 && key < 512) m_keys[key] = (action != 0);
    return true;
}

bool SimpleCamera::MousePosUpdate(double xpos, double ypos)
{
    float2 delta = float2(float(xpos), float(ypos)) - m_currentXY;
    if (m_pressedFlag)
    {
        m_cameraAzimuth -= delta.x * 0.01f;
        m_cameraElevation += delta.y * 0.01f;
        m_cameraElevation = std::max(m_cameraElevation, -1.57f + 0.01f);
        m_cameraElevation = std::min(m_cameraElevation, 1.57f - 0.01f);
    }
    else if (m_panFlag)
    {
        float3 offset = sphericalToCartesian(m_cameraAzimuth, m_cameraElevation, m_cameraDistance);
        float3 camDir = normalize(-offset);
        float3 camRight = normalize(cross(camDir, float3(0, 1, 0)));
        float3 camUp = cross(camRight, camDir);
        
        m_cameraTarget += camRight * delta.x * 0.005f * m_cameraDistance;
        m_cameraTarget += camUp * delta.y * 0.005f * m_cameraDistance;
    }
    m_currentXY = float2(float(xpos), float(ypos));
    return true;
}

bool SimpleCamera::MouseButtonUpdate(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) m_pressedFlag = (action != 0);
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) m_panFlag = (action != 0);
    return true;
}

bool SimpleCamera::MouseScrollUpdate(double xoffset, double yoffset)
{
    m_cameraDistance -= float(yoffset) * 0.5f;
    m_cameraDistance = std::max(m_cameraDistance, 0.1f);
    return true;
}

void SimpleCamera::Animate(float seconds)
{
    float3 offset = sphericalToCartesian(m_cameraAzimuth, m_cameraElevation, m_cameraDistance);
    float3 camDir = normalize(-offset);
    float3 camRight = normalize(cross(camDir, float3(0, 1, 0)));
    
    float moveSpeed = (m_keys[GLFW_KEY_LEFT_SHIFT] ? 3.0f : 1.0f) * seconds * 2.0f;
    if (m_keys[GLFW_KEY_W]) m_cameraTarget += camDir * moveSpeed;
    if (m_keys[GLFW_KEY_S]) m_cameraTarget -= camDir * moveSpeed;
    if (m_keys[GLFW_KEY_D]) m_cameraTarget -= camRight * moveSpeed;
    if (m_keys[GLFW_KEY_A]) m_cameraTarget += camRight * moveSpeed;
    if (m_keys[GLFW_KEY_E]) m_cameraTarget += float3(0, 1, 0) * moveSpeed;
    if (m_keys[GLFW_KEY_Q]) m_cameraTarget -= float3(0, 1, 0) * moveSpeed;
}

float3 SimpleCamera::GetPosition() const
{
    float3 offset = sphericalToCartesian(m_cameraAzimuth, m_cameraElevation, m_cameraDistance);
    return m_cameraTarget + offset;
}

float3 SimpleCamera::GetDir() const
{
    float3 offset = sphericalToCartesian(m_cameraAzimuth, m_cameraElevation, m_cameraDistance);
    return normalize(-offset);
}

float3 SimpleCamera::GetUp() const
{
    float3 dir = GetDir();
    float3 right = normalize(cross(dir, float3(0, 1, 0)));
    return cross(right, dir);
}

float3 SimpleCamera::GetRight() const
{
    float3 dir = GetDir();
    return normalize(cross(dir, float3(0, 1, 0)));
}
