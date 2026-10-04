#version 330 core

// Attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 in_normal;

uniform mat4 projection;
uniform mat4 modelview;
uniform mat4 model;

out vec3 WorldNormal;
out vec3 WorldPos;
out vec3 v_pos;

void main(){
    gl_Position = projection * modelview * vec4(position, 1.0);
    
    // World space normal (ignoring non-uniform scaling)
    WorldNormal = normalize(mat3(model) * in_normal);
    
    // Calculate and pass out the raw World Position
    WorldPos = vec3(model * vec4(position, 1.0));
    v_pos = WorldPos;
}