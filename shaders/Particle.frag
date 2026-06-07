#version 330 core

in vec3 color;
out vec4 fragColor;


void main()
{
    vec2 uv = gl_PointCoord * 2.0 - 1.0; // [-1, 1]

    float dist = dot(uv, uv);

    if (dist > 1.0)
        discard;

    vec3 color = vec3(0.2, 0.8, 0.9); // albo velocity itd.
    gl_FragColor = vec4(color, 1.0);
}
