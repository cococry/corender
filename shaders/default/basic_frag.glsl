#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout (location = 0) in vec4 v_color;
layout (location = 0) out vec4 o_color;

void main() {
  o_color = v_color;
}
