#ifndef SHAPE_INFERENCE_H
#define SHAPE_INFERENCE_H

#include "miniaic/ir/graph.h"

namespace opt {

// 对 graph 中的每个节点做 shape 推断（图优化的第一步，后续 pass 的前提）。
// 同时沿 shape 相关算子（Shape/Gather/Unsqueeze/Concat/Reshape）传播常量
// 数据，使 Reshape 等依赖常量输入的算子能推断出输出 shape。
// 返回 true 表示 graph 发生了变化。
bool shape_inference(Graph& graph);

}  // namespace opt

#endif  // SHAPE_INFERENCE_H
