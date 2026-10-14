"""GLSL shader sources (OpenGL 3.3 core profile).

The vertex shader applies the camera matrices; the fragment shader does Blinn-Phong
lighting with a 'headlight' that is fixed to the camera, so the model is always lit
from the viewer's side no matter how it is rotated.
"""

MESH_VS = """#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;   // inverse-transpose of the upper 3x3 of view * model

out vec3 v_pos_eye;             // position in eye (camera) space
out vec3 v_nrm_eye;             // normal   in eye space

void main() {
    vec4 pos_eye = u_view * u_model * vec4(in_position, 1.0);
    v_pos_eye   = pos_eye.xyz;
    v_nrm_eye   = u_normal_matrix * in_normal;
    gl_Position = u_proj * pos_eye;          // clip space; the GPU divides by w afterwards
}
"""

MESH_FS = """#version 330 core
in vec3 v_pos_eye;
in vec3 v_nrm_eye;

uniform vec3 u_color;
uniform bool u_flat;     // true: per-facet normal from screen-space derivatives
uniform bool u_unlit;    // true: plain colour (used for wireframe and point clouds)

out vec4 frag_color;

void main() {
    if (u_unlit) { frag_color = vec4(u_color, 1.0); return; }

    vec3 n;
    if (u_flat) {
        // Derivatives of the eye-space position across neighbouring pixels span the facet plane.
        n = normalize(cross(dFdx(v_pos_eye), dFdy(v_pos_eye)));
    } else {
        n = normalize(v_nrm_eye);
        if (!gl_FrontFacing) n = -n;         // two-sided lighting for open meshes
    }

    vec3 v    = normalize(-v_pos_eye);                 // towards the eye (the eye sits at the origin)
    vec3 key  = normalize(vec3(0.35, 0.55, 1.0));      // key light, slightly above-right of the camera
    vec3 fill = normalize(vec3(-0.6, -0.3, -0.5));     // weak fill light from the opposite side

    float diffuse  = 0.80 * max(dot(n, key), 0.0) + 0.20 * max(dot(n, fill), 0.0);
    vec3  h        = normalize(key + v);               // Blinn-Phong half vector
    float specular = 0.30 * pow(max(dot(n, h), 0.0), 48.0);

    vec3 color = u_color * (0.15 + diffuse) + vec3(specular);
    frag_color = vec4(color, 1.0);
}
"""

LINE_VS = """#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    v_color     = in_color;
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

LINE_FS = """#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() { frag_color = vec4(v_color, 1.0); }
"""
