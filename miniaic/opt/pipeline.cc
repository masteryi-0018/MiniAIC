#include "miniaic/opt/pipeline.h"

#include "miniaic/opt/constant_folding.h"
#include "miniaic/opt/fuse_conv_clip.h"
#include "miniaic/opt/shape_inference.h"

namespace opt {

PassManager build_default_pipeline() {
  PassManager passes;
  // shape 推断是后续 pass 的前提：融合/折叠/删除都依赖 tensor shape
  passes.add("shape_inference", shape_inference);
  // 常量折叠需要 shape，且折叠整条依赖链需要多次迭代
  passes.add("constant_folding", constant_folding, true);
  passes.add("fuse_conv_clip", fuse_conv_clip);
  return passes;
}

}  // namespace opt
