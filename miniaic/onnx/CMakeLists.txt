file(GLOB PROTO_SRC "*.cc")
add_library(proto ${LIB_TYPE} ${PROTO_SRC})
target_link_libraries(proto PRIVATE libprotobuf)

message(STATUS "Source files for proto library: ${PROTO_SRC}")