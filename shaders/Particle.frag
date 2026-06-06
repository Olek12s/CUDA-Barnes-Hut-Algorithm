#version 330 core

in vec3 color;
out vec4 fragColor;


void main()
{
    //fragColor = vec4(0f, 0.1f, 0.9f, 1.0f);
    fragColor = vec4(color, 1.0);
}
