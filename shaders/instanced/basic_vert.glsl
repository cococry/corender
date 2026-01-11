#version 450 

layout(location = 0) in vec2 i_pos;
layout(location = 1) in vec2 i_size;
layout(location = 2) in vec4 i_color;

layout (location = 0) flat out vec4 v_color;

layout(push_constant) uniform push_constant {
  vec2 ndc_to_px_scale;
  vec2 ndc_to_px_offset;
} pc;


const vec2 quad[6] = vec2[](
    vec2(0, 0),
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 0),
    vec2(1, 1),
    vec2(0, 1)
);


void main() {
  v_color = i_color;
  gl_Position = vec4((quad[gl_VertexIndex] * i_size + i_pos) * pc.ndc_to_px_scale + pc.ndc_to_px_offset, 0.0,1.0);
}
