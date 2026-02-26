cmake_minimum_required(VERSION 3.31)
project(Barnes-Hut-Licencjat LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_library(glad STATIC glad.c)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)
target_include_directories(glad PUBLIC "${CMAKE_SOURCE_DIR}/include")


add_executable(Barnes-Hut-Licencjat
        main.cpp
        cuda.cu
        src/Octtree.cpp
        include/Octtree.h
        include/Config.h
)


set_target_properties(Barnes-Hut-Licencjat PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


target_include_directories(Barnes-Hut-Licencjat PRIVATE "${CMAKE_SOURCE_DIR}/include")


target_link_libraries(Barnes-Hut-Licencjat PRIVATE
        glad
        "${CMAKE_SOURCE_DIR}/lib/glfw3.lib"
        opengl32
        gdi32
        user32
        shell32
)