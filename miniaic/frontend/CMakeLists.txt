file(GLOB MAIN_SRC "*.cc")
add_executable(main ${MAIN_SRC})

# should link protobuf library to main target, otherwise will get missing header error when compiling main.cc
target_link_libraries(main PRIVATE proto libprotobuf)