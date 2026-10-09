#ifndef CONSTANT_FOLDING_H
#define CONSTANT_FOLDING_H

#include "miniaic/ir/graph.h"

namespace opt {

// 常量折叠：把编译期就能确定结果的节点求值后删除，输出张量保留常量数据
// （效果等同 initializer，mininn 直接读它，不再执行该节点）。
//
// 两类可折叠节点：
//   1. 所有输入都是常量的算子：Add / Concat / Gather / Unsqueeze / Reshape
//   2. 仅依赖输入静态 shape 的 Shape —— 它的输入往往是运行时张量，但输出
//      只取决于静态 shape，因此同样可折叠
//
// 需要先跑 shape_inference（判断可折叠性与写入数据都要用 tensor shape）。
// 返回 true 表示图发生了变化。
bool constant_folding(Graph& graph);

}  // namespace opt

#endif  // CONSTANT_FOLDING_H
