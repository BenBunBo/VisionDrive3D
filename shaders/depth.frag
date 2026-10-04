#version 330 core

precision mediump float;

uniform float near;
uniform float far; 
uniform bool use_clip;
uniform vec2 clip_x;
uniform vec2 clip_z;

in vec3 v_pos;

out vec4 fragColor;

float linearize_depth(float depth){
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z *(far - near));
}

void main(){
    if (use_clip) {
        if (v_pos.x < clip_x.x || v_pos.x > clip_x.y ||
            v_pos.z < clip_z.x || v_pos.z > clip_z.y) {
            discard;
        }
    }

    float depth = gl_FragCoord.z;
    float linear_depth = linearize_depth(depth);

    float normalized_depth = linear_depth / far;

    fragColor = vec4(vec3(normalized_depth), 1.0);
}