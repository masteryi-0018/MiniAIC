file(GLOB IR_SRC "*.cc")
add_library(ir ${LIB_TYPE} ${IR_SRC})
target_link_libraries(ir PRIVATE miniaic_compile_options)

message(STATUS "Source files for ir library: ${IR_SRC}")