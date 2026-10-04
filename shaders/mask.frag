#version 330 core

precision mediump float;

uniform vec3 instance_color;
uniform bool use_clip;
uniform vec2 clip_x;
uniform vec2 clip_z;

in vec3 WorldNormal;
in vec3 WorldPos;
in vec3 v_pos;
out vec4 fragColor;

void main(){
    if (use_clip) {
        if (v_pos.x < clip_x.x || v_pos.x > clip_x.y ||
            v_pos.z < clip_z.x || v_pos.z > clip_z.y) {
            discard;
        }
    }

    if (instance_color.r < 0.0) {
        // Ground Condition: Pointing Upwards AND close to Floor Height (Y < 0.5)
        if (WorldNormal.y > 0.8 && v_pos.y < 0.5) {
            // Ground -> Yellow
            fragColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            // House/Roof/Walls -> Orange
            fragColor = vec4(1.0, 0.5, 0.0, 1.0);
        }
    } else {
        // Direct Semantic+Instance color from CPU
        fragColor = vec4(instance_color, 1.0);
    }
}