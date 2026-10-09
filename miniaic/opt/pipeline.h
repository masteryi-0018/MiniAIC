#ifndef PIPELINE_H
#define PIPELINE_H

#include "miniaic/opt/pass.h"

namespace opt {

// 默认图优化管线（注册顺序即执行顺序，前面的 pass 是后面的前提）
PassManager build_default_pipeline();

}  // namespace opt

#endif  // PIPELINE_H
