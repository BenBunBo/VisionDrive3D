#version 330 core

// Attributes (required by the project spec)
layout(location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 modelview;
uniform mat4 model;

out vec3 v_pos;

void main(){
    vec4 world = model * vec4(position, 1.0);
    v_pos = world.xyz;
    gl_Position = projection * modelview * vec4(position, 1.0);
}