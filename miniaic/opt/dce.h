#ifndef DCE_H
#define DCE_H

#include "miniaic/ir/graph.h"

namespace opt {

// 死代码消除：从 graph 输出反向标记活跃性，删除不活跃节点，并回收死张量
// （既不是 graph 输入输出、也不被活跃节点使用或产生）。
//
// 回收张量时需要重排 tensor 下标：graph 输入输出会同步重映射，而节点是通过
// 指针引用 tensor 的，不受下标变化影响。
//
// 所有算子都是纯函数（无副作用），删除不活跃节点是安全的。主要收益是运行时
// 内存：mininn 会为 .gynn 里每个 tensor 分配 buffer，融合/折叠留下的死张量
// （例如被融合掉的原 Conv 输出）会被真正回收。
//
// 返回 true 表示图发生了变化。
bool dce(Graph& graph);

}  // namespace opt

#endif  // DCE_H
