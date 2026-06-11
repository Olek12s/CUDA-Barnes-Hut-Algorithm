#version 330 core

in vec3 particleColor;
out vec4 fragColor;

void main()
{
    vec2 uv = gl_PointCoord * 2.0 - 1.0;

    float distSq = dot(uv, uv);
    if (distSq > 1.0) discard;

    float dist = sqrt(distSq);

    float glow = pow(1.0 - dist, 1.8);
    float core = smoothstep(0.4, 0.0, dist);

    vec3 finalColor = particleColor + (vec3(1.0) * core);

    fragColor = vec4(finalColor, glow);
}