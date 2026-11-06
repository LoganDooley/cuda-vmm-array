cmake_minimum_required(VERSION 3.21)
project(cuda_vmm_array VERSION 0.1.0 LANGUAGES CXX)

option(CUDA_VMM_ARRAY_BUILD_TESTS "Build tests?" ${PROJECT_IS_TOP_LEVEL})

find_package(CUDAToolkit REQUIRED)

add_library(cuda_vmm_array INTERFACE)
add_library(cuda_vmm_array::cuda_vmm_array ALIAS cuda_vmm_array)

target_include_directories(cuda_vmm_array INTERFACE
${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(cuda_vmm_array INTERFACE CUDA::cuda_driver)
target_compile_features(cuda_vmm_array INTERFACE cxx_std_17)

if(CUDA_VMM_ARRAY_BUILD_TESTS)
    enable_language(CUDA)
    enable_testing()

    add_executable(growable_device_array_test tests/growable_device_array_test.cu)
    target_link_libraries(growable_device_array_test PRIVATE
        cuda_vmm_array::cuda_vmm_array CUDA::cudart)
    set_target_properties(growable_device_array_test PROPERTIES
        CUDA_ARCHITECTURES native)

    add_test(NAME growable_device_array_test COMMAND growable_device_array_test)
endif()