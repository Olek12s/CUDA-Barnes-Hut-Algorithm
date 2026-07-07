cmake_minimum_required(VERSION 3.31)
project(Barnes-Hut-Licencjat LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

add_library(glad STATIC glad.c)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)
target_include_directories(glad PUBLIC "${CMAKE_SOURCE_DIR}/include")

file(GLOB IMGUI_SOURCES "${CMAKE_SOURCE_DIR}/include/imgui/*.cpp")

add_executable(Barnes-Hut-Licencjat
        main.cpp
        src/Octree.cpp
        include/Octree.h
        include/Globals.h
        include/Particle.h
        src/Renderer.cpp
        include/Renderer.h
        include/Shader.h
        src/Camera.cpp
        include/Camera.h
        ${IMGUI_SOURCES}
        include/ParticleGenerator.h
        src/ParticleGenerator.cpp
)

target_include_directories(Barnes-Hut-Licencjat PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/include/imgui"
)


target_include_directories(Barnes-Hut-Licencjat PRIVATE "${CMAKE_SOURCE_DIR}/include")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

    target_compile_options(Barnes-Hut-Licencjat PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

    target_compile_options(Barnes-Hut-Licencjat PRIVATE /O2 /arch:AVX2 /fp:fast)
endif()

find_package(OpenGL REQUIRED)
if(WIN32)
    target_link_libraries(Barnes-Hut-Licencjat PRIVATE
            glad
            "${CMAKE_SOURCE_DIR}/lib/glfw3.lib"
            opengl32
            gdi32
            user32
            shell32
    )
elseif(UNIX)
    set(glfw3_DIR "/usr/lib64/cmake/glfw3")
    find_package(glfw3 CONFIG REQUIRED)

    target_link_libraries(Barnes-Hut-Licencjat PRIVATE
            glad
            glfw
            OpenGL::GL
    )
endif()