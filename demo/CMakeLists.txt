file(GLOB DEMO_SRC "*.cc")
add_executable(demo ${DEMO_SRC})

message(STATUS "Source files for demo binary: ${DEMO_SRC}")

# MiniNN 头文件（写死路径）
include_directories("/home/masteryi/work/MiniNN")
include_directories("${MINIAIC_SOURCE_DIR}/third_party/flatbuffers/include")

# MiniNN 静态库（写死路径，直接指向 MiniNN 构建产物）
target_link_libraries(demo
  /home/masteryi/work/MiniNN/build/mininn/libmininn_lib.a
  /home/masteryi/work/MiniNN/build/mininn/backend/avx/libavx.a
  /home/masteryi/work/MiniNN/build/mininn/backend/sse/libsse.a
)
if(UNIX AND NOT ANDROID)
  target_link_libraries(demo pthread)
endif()

install(TARGETS demo RUNTIME DESTINATION bin)
