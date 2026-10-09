#ifndef DCE_H
#define DCE_H

#include "miniaic/ir/graph.h"
#include "miniaic/opt/pass.h"

namespace opt {

// 死代码消除（Dead Code Elimination）：
//   1. 从 graph 输出反向传播活跃性，删除输出不被任何活跃节点/输出使用的节点
//   2. 回收死张量（既不是 graph 输入输出、也不被活跃节点使用或产生），并
//      重排 tensor 下标（graph 输入输出同步重映射；节点通过指针引用 tensor，
//      不受下标变化影响）
//
// 所有算子都是纯函数（无副作用），因此删除不活跃节点是安全的。
// 主要收益是**运行时内存**：mininn 会为 .gynn 里每个 tensor 分配 buffer，
// 融合/折叠留下的死张量（例如被融合掉的原 Conv 输出）会被真正回收。
//
// count = 删除的节点数 + 回收的张量数。
PassResult dce(Graph& graph);

}  // namespace opt

#endif  // DCE_H
