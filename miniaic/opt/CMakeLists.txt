file(GLOB OPT_SRC "*.cc")
add_library(opt ${LIB_TYPE} ${OPT_SRC})
target_link_libraries(opt PRIVATE miniaic_compile_options)

message(STATUS "Source files for opt library: ${OPT_SRC}")
