#ifndef CONSTANT_FOLDING_H
#define CONSTANT_FOLDING_H

#include "miniaic/ir/graph.h"
#include "miniaic/opt/pass.h"

namespace opt {

// 常量折叠：把编译期就能确定结果的节点在编译期求值，写入其输出张量后删除
// 该节点（输出张量保留常量数据，效果等同 initializer）。
//
// 可折叠的两类节点：
//   1. 所有输入都是常量的可求值算子：Add / Clip / Concat / Gather /
//      Unsqueeze / Reshape（纯搬运数据）
//   2. 仅依赖输入静态 shape 的 Shape —— 它的输入往往是运行时张量，但输出
//      只取决于静态 shape，因此同样可折叠
//
// 前提：shape 已推断完（依赖 tensor shape 判断可折叠性与写入数据尺寸）。
// 为了保证一次执行能折叠整条依赖链，管线里以 fixpoint 方式运行。
//
// count = 本次折叠掉的节点数。
PassResult constant_folding(Graph& graph);

}  // namespace opt

#endif  // CONSTANT_FOLDING_H
