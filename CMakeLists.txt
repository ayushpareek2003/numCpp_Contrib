cmake_minimum_required(VERSION 3.22)

project(NUMC LANGUAGES CUDA)

find_package(CUDA REQUIRED)

include_directories(include)

set(SOURCES main.cu
    src/numC/gpuConfig.cu
)

add_executable(main ${SOURCES})
set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main cublas)