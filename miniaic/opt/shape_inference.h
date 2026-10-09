#ifndef SHAPE_INFERENCE_H
#define SHAPE_INFERENCE_H

#include "miniaic/ir/graph.h"
#include "miniaic/opt/pass.h"

namespace opt {

// 对 graph 中的每个节点做 shape 推断（图优化的第一步）。
// 同时沿 shape 相关算子（Shape/Gather/Unsqueeze/Concat/Reshape）传播常量
// 数据，使 Reshape 等依赖常量输入的算子能推断出输出 shape。
// count = 输出 shape 被更新的节点数。
PassResult shape_inference(Graph& graph);

}  // namespace opt

#endif  // SHAPE_INFERENCE_H
