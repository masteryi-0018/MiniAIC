#ifndef FUSE_CONV_CLIP_H
#define FUSE_CONV_CLIP_H

#include "miniaic/ir/graph.h"
#include "miniaic/opt/pass.h"

namespace opt {

// 将 Conv 后紧跟的 Clip 融合进 Conv：把 Clip 的 min/max 合并为 Conv 的可选
// 属性（MiniNN 侧 Conv 支持可选 clip），把 Conv 的输出重连为 Clip 的输出，
// 并删除 Clip 节点（tensor 不回收，下标保持不变）。
// 只在满足以下条件时融合，否则跳过：
//   1. Clip 的输入由某个 Conv 产生
//   2. 该 Conv 输出只被这一个 Clip 消费（否则不能重连）
//   3. 该 Conv 输出本身不是 graph 输出
//   4. Clip 同时有非空的 min 和 max
// 返回 count = 融合掉的 Conv+Clip 对数。
PassResult fuse_conv_clip(Graph& graph);

}  // namespace opt

#endif  // FUSE_CONV_CLIP_H
