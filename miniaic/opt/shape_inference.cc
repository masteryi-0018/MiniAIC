#include "miniaic/opt/shape_inference.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "miniaic/ir/node.h"
#include "miniaic/ir/tensor.h"
#include "miniaic/ir/type.h"

namespace opt {
namespace {

using TensorPtr = std::shared_ptr<Tensor>;

// ---- 通用工具 ----

// 未知 shape：注册时用 {-1} 标记，由 shape_inference 填充
bool UnknownShape(const TensorPtr& t) {
  const std::vector<int>& s = t->get_shape();
  return s.size() == 1 && s[0] == -1;
}

// 已知 shape：不是未知标记就算已知（空 shape 表示标量，也算已知）
bool KnownShape(const TensorPtr& t) { return !UnknownShape(t); }

bool SetShape(const TensorPtr& t, const std::vector<int>& shape) {
  if (t->get_shape() == shape) {
    return false;
  }
  std::vector<int> tmp = shape;
  t->set_shape(tmp);
  return true;
}

// 写入数据（大小不匹配时忽略）
void SetData(const TensorPtr& t, const std::vector<float>& data) {
  if (t->get_size() != static_cast<int>(data.size())) {
    return;
  }
  std::vector<float> tmp = data;
  t->set_data(tmp);
}

int GetAttrInt(const std::map<std::string, std::vector<int>>& attrs,
               const std::string& key, int default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
    return default_value;
  }
  return it->second[0];
}

std::vector<int> GetAttrInts(
    const std::map<std::string, std::vector<int>>& attrs,
    const std::string& key) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    return {};
  }
  return it->second;
}

// Numpy 式广播：两个 shape 对齐后，维度相等或其一为 1
bool BroadcastShape(const std::vector<int>& a, const std::vector<int>& b,
                    std::vector<int>* out) {
  std::vector<int> result(std::max(a.size(), b.size()), 1);
  int i = static_cast<int>(a.size()) - 1;
  int j = static_cast<int>(b.size()) - 1;
  int k = static_cast<int>(result.size()) - 1;
  while (k >= 0) {
    const int da = i >= 0 ? a[i] : 1;
    const int db = j >= 0 ? b[j] : 1;
    if (da != db && da != 1 && db != 1) {
      return false;
    }
    result[k] = std::max(da, db);
    --i;
    --j;
    --k;
  }
  *out = result;
  return true;
}

// ---- 各算子的 shape 推断 ----

// Clip 等输出 shape 与输入一致的算子
bool InferCopy(const std::vector<TensorPtr>& in,
               const std::vector<TensorPtr>& out) {
  if (in.empty() || out.empty() || !KnownShape(in[0])) {
    return false;
  }
  return SetShape(out[0], in[0]->get_shape());
}

bool InferAdd(const std::vector<TensorPtr>& in,
              const std::vector<TensorPtr>& out) {
  if (in.size() < 2 || out.empty() || !KnownShape(in[0]) ||
      !KnownShape(in[1])) {
    return false;
  }
  std::vector<int> shape;
  if (!BroadcastShape(in[0]->get_shape(), in[1]->get_shape(), &shape)) {
    return false;
  }
  return SetShape(out[0], shape);
}

bool InferConv(const std::vector<TensorPtr>& in,
               const std::vector<TensorPtr>& out,
               const std::map<std::string, std::vector<int>>& attrs) {
  if (in.size() < 2 || out.empty()) {
    return false;
  }
  const std::vector<int>& input_shape = in[0]->get_shape();
  const std::vector<int>& weight_shape = in[1]->get_shape();
  if (input_shape.size() < 4 || weight_shape.size() < 4) {
    return false;
  }
  std::vector<int> kernel = GetAttrInts(attrs, "kernel_shape");
  if (kernel.empty()) {
    kernel = {weight_shape[2], weight_shape[3]};
  }
  std::vector<int> pads = GetAttrInts(attrs, "pads");
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  std::vector<int> strides = GetAttrInts(attrs, "strides");
  if (strides.empty()) {
    strides = {1, 1};
  }
  std::vector<int> dilations = GetAttrInts(attrs, "dilations");
  if (dilations.empty()) {
    dilations = {1, 1};
  }
  const int out_h =
      (input_shape[2] + pads[0] + pads[2] - dilations[0] * (kernel[0] - 1) -
       1) /
          strides[0] +
      1;
  const int out_w =
      (input_shape[3] + pads[1] + pads[3] - dilations[1] * (kernel[1] - 1) -
       1) /
          strides[1] +
      1;
  std::vector<int> shape = {input_shape[0], weight_shape[0], out_h, out_w};
  return SetShape(out[0], shape);
}

// Shape: 输出 shape = [rank]，输出数据 = 输入各维的值
bool InferShapeOp(const std::vector<TensorPtr>& in,
                  const std::vector<TensorPtr>& out) {
  if (in.empty() || out.empty() || !KnownShape(in[0])) {
    return false;
  }
  const std::vector<int>& s = in[0]->get_shape();
  const bool changed = SetShape(out[0], {static_cast<int>(s.size())});
  std::vector<float> data(s.begin(), s.end());
  SetData(out[0], data);
  return changed;
}

// Gather: 输出 shape = 输入去掉/替换 axis 维；indices 为标量时传播数据
bool InferGather(const std::vector<TensorPtr>& in,
                 const std::vector<TensorPtr>& out,
                 const std::map<std::string, std::vector<int>>& attrs) {
  if (in.size() < 2 || out.empty() || !KnownShape(in[0])) {
    return false;
  }
  const std::vector<int>& s = in[0]->get_shape();
  int axis = GetAttrInt(attrs, "axis", 0);
  if (axis < 0) {
    axis += static_cast<int>(s.size());
  }
  if (axis < 0 || axis >= static_cast<int>(s.size())) {
    return false;
  }
  // indices 的 shape（未知时按标量处理）
  std::vector<int> indices_shape;
  if (KnownShape(in[1])) {
    indices_shape = in[1]->get_shape();
  }
  std::vector<int> shape;
  for (int i = 0; i < axis; ++i) {
    shape.push_back(s[i]);
  }
  shape.insert(shape.end(), indices_shape.begin(), indices_shape.end());
  for (int i = axis + 1; i < static_cast<int>(s.size()); ++i) {
    shape.push_back(s[i]);
  }
  const bool changed = SetShape(out[0], shape);

  // 数据传播：indices 为标量（1 个元素）时，取 axis 维上的一个切片
  if (in[0]->has_data() && in[1]->has_data() && in[1]->get_size() == 1) {
    const std::vector<float> input_data = in[0]->get_data();
    const std::vector<float> index_data = in[1]->get_data();
    int axis_stride = 1;
    for (int i = axis + 1; i < static_cast<int>(s.size()); ++i) {
      axis_stride *= s[i];
    }
    int idx = static_cast<int>(index_data[0]);
    if (idx < 0) {
      idx += s[axis];
    }
    const int in_stride = axis_stride * s[axis];
    const int out_size = in[0]->get_size() / s[axis];
    std::vector<float> data;
    data.reserve(out_size);
    for (int p = 0; p < out_size; ++p) {
      const int block = p / axis_stride;
      const int offset = p % axis_stride;
      data.push_back(input_data[block * in_stride + idx * axis_stride + offset]);
    }
    SetData(out[0], data);
  }
  return changed;
}

// Unsqueeze: 在 axes 位置插入 1，数据不变
bool InferUnsqueeze(const std::vector<TensorPtr>& in,
                    const std::vector<TensorPtr>& out,
                    const std::map<std::string, std::vector<int>>& attrs) {
  if (in.empty() || out.empty() || !KnownShape(in[0])) {
    return false;
  }
  std::vector<int> axes = GetAttrInts(attrs, "axes");
  if (axes.empty()) {
    return false;
  }
  std::vector<int> shape = in[0]->get_shape();
  for (int ax : axes) {
    const int pos =
        ax < 0 ? ax + static_cast<int>(shape.size()) + 1 : ax;
    if (pos < 0 || pos > static_cast<int>(shape.size())) {
      return false;
    }
    shape.insert(shape.begin() + pos, 1);
  }
  const bool changed = SetShape(out[0], shape);
  if (in[0]->has_data()) {
    SetData(out[0], in[0]->get_data());
  }
  return changed;
}

// Concat: 在 axis 维求和；axis==0 时传播数据（顺序拼接）
bool InferConcat(const std::vector<TensorPtr>& in,
                 const std::vector<TensorPtr>& out,
                 const std::map<std::string, std::vector<int>>& attrs) {
  if (in.empty() || out.empty()) {
    return false;
  }
  for (const auto& t : in) {
    if (!KnownShape(t)) {
      return false;
    }
  }
  std::vector<int> shape = in[0]->get_shape();
  int axis = GetAttrInt(attrs, "axis", 0);
  if (axis < 0) {
    axis += static_cast<int>(shape.size());
  }
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    return false;
  }
  int concat_dim = 0;
  for (const auto& t : in) {
    concat_dim += t->get_shape()[axis];
  }
  shape[axis] = concat_dim;
  const bool changed = SetShape(out[0], shape);
  if (axis == 0) {
    bool all_data = true;
    for (const auto& t : in) {
      if (!t->has_data()) {
        all_data = false;
        break;
      }
    }
    if (all_data) {
      std::vector<float> data;
      for (const auto& t : in) {
        const std::vector<float> d = t->get_data();
        data.insert(data.end(), d.begin(), d.end());
      }
      SetData(out[0], data);
    }
  }
  return changed;
}

// GlobalAveragePool: [N, C, H, W] -> [N, C, 1, 1]
bool InferGAP(const std::vector<TensorPtr>& in,
              const std::vector<TensorPtr>& out) {
  if (in.empty() || out.empty() || !KnownShape(in[0])) {
    return false;
  }
  const std::vector<int>& s = in[0]->get_shape();
  if (s.size() < 2) {
    return false;
  }
  return SetShape(out[0], {s[0], s[1], 1, 1});
}

// Reshape: 目标 shape 来自第二个输入（shape 常量）的数据
bool InferReshape(const std::vector<TensorPtr>& in,
                  const std::vector<TensorPtr>& out) {
  if (in.size() < 2 || out.empty() || !in[1]->has_data()) {
    return false;
  }
  const std::vector<float> shape_data = in[1]->get_data();
  std::vector<int> target;
  int unknown_idx = -1;
  long total_known = 1;
  for (size_t i = 0; i < shape_data.size(); ++i) {
    const int d = static_cast<int>(shape_data[i]);
    if (d == -1) {
      unknown_idx = static_cast<int>(i);
      target.push_back(-1);
    } else {
      total_known *= d;
      target.push_back(d);
    }
  }
  if (unknown_idx >= 0) {
    if (total_known == 0 || in[0]->get_size() == 0) {
      return false;
    }
    target[unknown_idx] = in[0]->get_size() / total_known;
  }
  const bool changed = SetShape(out[0], target);
  if (in[0]->has_data()) {
    SetData(out[0], in[0]->get_data());  // reshape 不改变数据
  }
  return changed;
}

// Gemm: A[M,K] x B -> [M, N]（transB 决定 N 取自 B 的第 0 或第 1 维）
bool InferGemm(const std::vector<TensorPtr>& in,
               const std::vector<TensorPtr>& out,
               const std::map<std::string, std::vector<int>>& attrs) {
  if (in.size() < 2 || out.empty()) {
    return false;
  }
  if (!KnownShape(in[0]) || !KnownShape(in[1])) {
    return false;
  }
  const std::vector<int>& a = in[0]->get_shape();
  const std::vector<int>& b = in[1]->get_shape();
  if (a.size() < 2 || b.size() < 2) {
    return false;
  }
  const int trans_b = GetAttrInt(attrs, "transB", 0);
  const int m = a[0];
  const int n = trans_b ? b[0] : b[1];
  return SetShape(out[0], {m, n});
}

// Constant: 输出 shape 已在 parser 阶段从 value tensor 注册
bool InferConstant(const std::vector<TensorPtr>& in,
                   const std::vector<TensorPtr>& out) {
  (void)in;
  (void)out;
  return false;
}

bool InferNode(const std::shared_ptr<Node>& node) {
  const Op op = node->get_op_type();
  const std::vector<TensorPtr>& in = node->get_input_tensors();
  const std::vector<TensorPtr>& out = node->get_output_tensors();
  const std::map<std::string, std::vector<int>>& attrs =
      node->get_attributes();
  switch (op) {
    case Op::ADD:
      return InferAdd(in, out);
    case Op::CONV:
      return InferConv(in, out, attrs);
    case Op::CLIP:
      return InferCopy(in, out);
    case Op::SHAPE:
      return InferShapeOp(in, out);
    case Op::GATHER:
      return InferGather(in, out, attrs);
    case Op::UNSQUEEZE:
      return InferUnsqueeze(in, out, attrs);
    case Op::CONCAT:
      return InferConcat(in, out, attrs);
    case Op::GLOBALAVERAGEPOOL:
      return InferGAP(in, out);
    case Op::RESHAPE:
      return InferReshape(in, out);
    case Op::GEMM:
      return InferGemm(in, out, attrs);
    case Op::CONSTANT:
      return InferConstant(in, out);
  }
  return false;
}

}  // namespace

bool shape_inference(Graph& graph) {
  int updated = 0;
  // onnx 的 node 顺序即为拓扑序，单遍即可
  for (const auto& node : graph.get_nodes()) {
    if (InferNode(node)) {
      ++updated;
    }
  }
  if (updated > 0) {
    std::cout << "Shape inference: " << updated << " node(s) updated"
              << std::endl;
  }
  return updated > 0;
}

}  // namespace opt
