#include "SimpleCamera.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

using namespace donut::math;

SimpleCamera::SimpleCamera()
{
    // Start by looking at origin from (0,0,2)
    m_position = float3(0.f, 0.f, 2.f);
    m_yaw = -3.14159265f / 2.0f; // -90 degrees in radians so looking down -Z
    m_pitch = 0.f;
    UpdateVectors();
}

void SimpleCamera::UpdateVectors()
{
    float3 forward;
    forward.x = std::cos(m_yaw) * std::cos(m_pitch);
    forward.y = std::sin(m_pitch);
    forward.z = std::sin(m_yaw) * std::cos(m_pitch);
    
    m_forward = normalize(forward);
    m_right = normalize(cross(float3(0.f, 1.f, 0.f), m_forward));
    m_up = normalize(cross(m_forward, m_right));
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
        float sensitivity = 0.005f;
        m_yaw -= delta.x * sensitivity;
        m_pitch -= delta.y * sensitivity;

        // constrain pitch
        m_pitch = std::max(m_pitch, -1.57f + 0.01f);
        m_pitch = std::min(m_pitch, 1.57f - 0.01f);

        UpdateVectors();
    }
    m_currentXY = float2(float(xpos), float(ypos));
    return true;
}

bool SimpleCamera::MouseButtonUpdate(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) m_pressedFlag = (action != 0);
    return true;
}

bool SimpleCamera::MouseScrollUpdate(double xoffset, double yoffset)
{
    // Move forward/backward via scroll
    m_position += m_forward * float(yoffset) * 0.5f;
    return true;
}

void SimpleCamera::Animate(float seconds)
{
    float moveSpeed = (m_keys[GLFW_KEY_LEFT_SHIFT] ? 3.0f : 1.0f) * seconds * 2.0f;
    if (m_keys[GLFW_KEY_W]) m_position += m_forward * moveSpeed;
    if (m_keys[GLFW_KEY_S]) m_position -= m_forward * moveSpeed;
    if (m_keys[GLFW_KEY_D]) m_position += m_right * moveSpeed;
    if (m_keys[GLFW_KEY_A]) m_position -= m_right * moveSpeed;
    if (m_keys[GLFW_KEY_E]) m_position += float3(0, 1, 0) * moveSpeed;
    if (m_keys[GLFW_KEY_Q]) m_position -= float3(0, 1, 0) * moveSpeed;
}

float3 SimpleCamera::GetPosition() const
{
    return m_position;
}

float3 SimpleCamera::GetDir() const
{
    return m_forward;
}

float3 SimpleCamera::GetUp() const
{
    return m_up;
}

float3 SimpleCamera::GetRight() const
{
    return m_right;
}
