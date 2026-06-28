file(GLOB MAIN_SRC "*.cc")
file(GLOB PROTO_SRC "onnx/*.cc")

set(PROTO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${PROTO_SOURCE_DIR}")

add_executable(main ${MAIN_SRC} ${PROTO_SRC})

# should link protobuf library to main target, otherwise will get missing header error when compiling main.cc
target_link_libraries(main PRIVATE protobuf::libprotobuf)