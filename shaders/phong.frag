#version 330 core

precision mediump float;

in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;
in vec2 v_uv;

uniform vec3 light_pos;
uniform vec3 light_color;
uniform float shininess;
uniform float ambient_strength;
uniform float specular_strength;
uniform sampler2D diffuse_map;
uniform int has_texture;
uniform int use_uniform_base_color;
uniform vec3 base_color;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_pos - v_pos);
    vec3 R = reflect(-L, N);
    vec3 V = normalize(-v_pos);

    vec3 surface_color = v_color;
    if (has_texture != 0) {
        surface_color = texture(diffuse_map, v_uv).rgb;
    } else if (use_uniform_base_color != 0) {
        surface_color = base_color;
    }

    float diffuse = max(dot(L, N), 0.0);
    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);

    vec3 ambient = ambient_strength * surface_color * light_color;
    vec3 diffuse_term = diffuse * surface_color * light_color;
    vec3 specular_term = specular_strength * specular * light_color;
    vec3 rgb = ambient + diffuse_term + specular_term;
    fragColor = vec4(rgb, 1.0);
}

