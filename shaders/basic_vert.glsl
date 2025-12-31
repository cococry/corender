#version 450 

layout (location = 0) in vec2 a_pos;

layout (location = 1) in vec2 i_pos;
layout (location = 2) in vec2 i_size;
layout (location = 3) in vec4 i_color;

layout (location = 0) out vec4 v_color;

layout(push_constant) uniform push_constant {
  vec2 ndc_to_px_scale;
  vec2 ndc_to_px_offset;
} pc;

void main() {
  v_color = i_color;
  gl_Position = vec4(
      (a_pos * i_size + i_pos) * pc.ndc_to_px_scale + pc.ndc_to_px_offset, 0.0, 1.0);

}
