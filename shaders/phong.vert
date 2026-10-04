#version 330 core

// Attributes (required by the project spec)
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 color;

uniform mat4 projection;
uniform mat4 modelview;

out vec3 v_normal;
out vec3 v_pos;
out vec3 v_color;
out vec2 v_uv;

void main() {
    vec4 pos4 = modelview * vec4(position, 1.0);
    v_pos = vec3(pos4) / pos4.w;

    mat4 normal_matrix = transpose(inverse(modelview));
    v_normal = normalize(vec3(normal_matrix * vec4(normal, 0.0)));

    v_color = color;
    v_uv = texcoord;
    gl_Position = projection * pos4;
}

