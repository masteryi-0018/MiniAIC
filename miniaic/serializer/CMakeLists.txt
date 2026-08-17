file(GLOB SERIALIZER_SRC "*.cc")
add_library(serializer ${LIB_TYPE} ${SERIALIZER_SRC})
target_link_libraries(serializer PRIVATE miniaic_compile_options)

message(STATUS "Source files for serializer library: ${SERIALIZER_SRC}")
