#include "miniaic/opt/pipeline.h"

#include "miniaic/opt/fuse_conv_clip.h"
#include "miniaic/opt/shape_inference.h"

namespace opt {

PassManager build_default_pipeline() {
  PassManager passes;
  // shape 推断是后续 pass 的前提：融合/折叠/删除都依赖 tensor shape
  passes.add("shape_inference", shape_inference);
  passes.add("fuse_conv_clip", fuse_conv_clip);
  return passes;
}

}  // namespace opt
