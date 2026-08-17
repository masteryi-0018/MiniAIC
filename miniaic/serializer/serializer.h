#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <string>

#include "miniaic/ir/graph.h"

namespace serializer {

// 将 IR graph 序列化为 MiniNN 可读的 .gynn flatbuffer 文件。
// 返回 true 表示成功。
bool save(Graph& graph, const std::string& path);

}  // namespace serializer

#endif  // SERIALIZER_H
