# 1. environment
# for miniaic/onnx/onnx.pb.cc to compile successfully, need to include the current source directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# pybind need one static lib to produce a shared lib, so use a OBJECT to handle it
set(LIB_TYPE OBJECT) # options: OBJECT, STATIC, SHARED

# use interface library to manage compile options
add_library(miniaic_compile_options INTERFACE)

# set strict compile options for the interface library
target_compile_options(miniaic_compile_options INTERFACE
    -Werror
    -Wall
    -Wextra
    -Wpedantic
)

# 2. subdirectory
add_subdirectory(frontend)
add_subdirectory(ir)
add_subdirectory(onnx)