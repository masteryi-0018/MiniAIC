#include "miniaic/opt/constant_folding.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "miniaic/ir/node.h"
#include "miniaic/ir/tensor.h"
#include "miniaic/ir/type.h"

namespace opt {
namespace {

// shape 未知（shape_inference 留下的标记）
bool ShapeUnknown(const std::vector<int>& shape) {
  return shape.size() == 1 && shape[0] == -1;
}

// 元素个数（空 shape 表示标量，1 个元素）
int NumElements(const std::vector<int>& shape) {
  int size = 1;
  for (int dim : shape) {
    size *= dim;
  }
  return size;
}

int GetAttrInt(const std::map<std::string, std::vector<int>>& attrs,
               const std::string& key, int default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
    return default_value;
  }
  return it->second[0];
}

// 取常量 tensor 的数据
std::vector<float> ReadData(const std::shared_ptr<Tensor>& tensor) {
  return tensor->get_data();
}

// Shape：输出 = 输入 shape 的各维
std::vector<float> EvalShape(const std::vector<int>& input_shape) {
  return std::vector<float>(input_shape.begin(), input_shape.end());
}

// Add：逐元素相加（要求两个输入形状完全一致，与 mininn 的 add kernel 一致）
bool EvalAdd(const std::vector<float>& a, const std::vector<float>& b,
             std::vector<float>* out) {
  if (a.size() != b.size()) {
    return false;
  }
  out->resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    (*out)[i] = a[i] + b[i];
  }
  return true;
}

// Clip：按 [min, max] 截断（边界来自属性，与 mininn 的 clip kernel 一致）
bool EvalClip(const std::vector<float>& data, int min_value, int max_value,
              std::vector<float>* out) {
  out->resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] > max_value) {
      (*out)[i] = static_cast<float>(max_value);
    } else if (data[i] < min_value) {
      (*out)[i] = static_cast<float>(min_value);
    } else {
      (*out)[i] = data[i];
    }
  }
  return true;
}

// Concat：沿 axis 拼接（各输入除 axis 外形状必须一致）
bool EvalConcat(const std::vector<std::vector<float>>& datas,
                const std::vector<std::vector<int>>& shapes, int axis,
                std::vector<float>* out) {
  if (datas.empty() || datas.size() != shapes.size()) {
    return false;
  }
  const int rank = static_cast<int>(shapes[0].size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return false;
  }
  int outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= shapes[0][i];
  }
  int inner = 1;
  for (int i = axis + 1; i < rank; ++i) {
    inner *= shapes[0][i];
  }
  for (size_t k = 0; k < datas.size(); ++k) {
    if (shapes[k].size() != shapes[0].size()) {
      return false;
    }
    for (int i = 0; i < rank; ++i) {
      if (i != axis && shapes[k][i] != shapes[0][i]) {
        return false;
      }
    }
    if (static_cast<int>(datas[k].size()) != NumElements(shapes[k])) {
      return false;
    }
  }
  out->clear();
  for (int o = 0; o < outer; ++o) {
    for (size_t k = 0; k < datas.size(); ++k) {
      const int dim = shapes[k][axis];
      const size_t begin = static_cast<size_t>(o) * dim * inner;
      out->insert(out->end(), datas[k].begin() + begin,
                  datas[k].begin() + begin + dim * inner);
    }
  }
  return true;
}

// Gather：沿 axis 按下标取数据
bool EvalGather(const std::vector<float>& data,
                const std::vector<int>& data_shape,
                const std::vector<float>& indices, int axis,
                std::vector<float>* out) {
  const int rank = static_cast<int>(data_shape.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank || static_cast<int>(data.size()) !=
                                      NumElements(data_shape)) {
    return false;
  }
  int axis_stride = 1;
  for (int i = axis + 1; i < rank; ++i) {
    axis_stride *= data_shape[i];
  }
  int outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= data_shape[i];
  }
  const int in_stride = axis_stride * data_shape[axis];
  out->clear();
  out->reserve(static_cast<size_t>(outer) * indices.size() * axis_stride);
  for (int o = 0; o < outer; ++o) {
    for (float index : indices) {
      int idx = static_cast<int>(index);
      if (idx < 0) {
        idx += data_shape[axis];
      }
      if (idx < 0 || idx >= data_shape[axis]) {
        return false;
      }
      const size_t begin = static_cast<size_t>(o) * in_stride + idx * axis_stride;
      out->insert(out->end(), data.begin() + begin,
                  data.begin() + begin + axis_stride);
    }
  }
  return true;
}

// 按算子求值；成功返回 true 并填充 out
bool EvalNode(const std::shared_ptr<Node>& node, std::vector<float>* out) {
  const Op op = node->get_op_type();
  const std::vector<std::shared_ptr<Tensor>>& in = node->get_input_tensors();
  const std::map<std::string, std::vector<int>>& attrs = node->get_attributes();
  switch (op) {
    case Op::ADD:
      return in.size() == 2 &&
             in[0]->get_shape() == in[1]->get_shape() &&
             EvalAdd(ReadData(in[0]), ReadData(in[1]), out);
    case Op::CLIP:
      return !in.empty() &&
             EvalClip(ReadData(in[0]), GetAttrInt(attrs, "min", 0),
                      GetAttrInt(attrs, "max", 0), out);
    case Op::CONCAT: {
      std::vector<std::vector<float>> datas;
      std::vector<std::vector<int>> shapes;
      for (const auto& tensor : in) {
        datas.push_back(ReadData(tensor));
        shapes.push_back(tensor->get_shape());
      }
      return EvalConcat(datas, shapes, GetAttrInt(attrs, "axis", 0), out);
    }
    case Op::GATHER:
      return in.size() == 2 &&
             EvalGather(ReadData(in[0]), in[0]->get_shape(), ReadData(in[1]),
                        GetAttrInt(attrs, "axis", 0), out);
    case Op::UNSQUEEZE:
    case Op::RESHAPE:
      // 纯搬运数据，数据不变
      if (in.empty()) {
        return false;
      }
      *out = ReadData(in[0]);
      return true;
    default:
      return false;
  }
}

}  // namespace

PassResult constant_folding(Graph& graph) {
  std::vector<std::shared_ptr<Node>>& nodes = graph.get_nodes();
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();

  // tensor -> 下标
  std::map<const Tensor*, int> index_of;
  for (size_t i = 0; i < tensors.size(); ++i) {
    index_of[tensors[i].get()] = static_cast<int>(i);
  }

  // 常量张量集合：initializer（有数据且没有生产者节点）+ Constant 节点的输出
  std::set<const Tensor*> produced;
  for (const auto& node : nodes) {
    for (const auto& tensor : node->get_output_tensors()) {
      produced.insert(tensor.get());
    }
  }
  std::set<const Tensor*> constant;
  for (const auto& tensor : tensors) {
    if (tensor->has_data() && produced.count(tensor.get()) == 0) {
      constant.insert(tensor.get());
    }
  }
  for (const auto& node : nodes) {
    if (node->get_op_type() == Op::CONSTANT) {
      for (const auto& tensor : node->get_output_tensors()) {
        constant.insert(tensor.get());
      }
    }
  }

  std::set<int> graph_outputs;
  for (int index : graph.get_outputs()) {
    graph_outputs.insert(index);
  }

  PassResult result;
  std::vector<std::shared_ptr<Node>> folded;
  for (const auto& node : nodes) {
    const Op op = node->get_op_type();
    if (op == Op::CONSTANT) {
      continue;  // 常量生产者本身不需要折叠（无人使用由 DCE 处理）
    }
    const std::vector<std::shared_ptr<Tensor>>& in = node->get_input_tensors();
    const std::vector<std::shared_ptr<Tensor>>& out = node->get_output_tensors();
    if (in.empty() || out.size() != 1) {
      continue;  // 只折叠单输出算子
    }
    // 输出是 graph 输出时保留其生产者，维持"graph 输出必有生产者"
    if (graph_outputs.count(index_of[out[0].get()]) > 0) {
      continue;
    }

    std::vector<float> data;
    bool foldable = false;
    if (op == Op::SHAPE) {
      // 只依赖输入的静态 shape：输入是运行时张量也可折叠
      const std::vector<int>& shape = in[0]->get_shape();
      if (!shape.empty() && !ShapeUnknown(shape)) {
        data = EvalShape(shape);
        foldable = true;
      }
    } else {
      bool all_constant = true;
      for (const auto& tensor : in) {
        if (constant.count(tensor.get()) == 0) {
          all_constant = false;
        }
      }
      foldable = all_constant && EvalNode(node, &data);
    }
    if (!foldable) {
      continue;
    }
    // 数据尺寸必须与输出张量一致（形状未知时尺寸也是 1，会在此被挡掉）
    if (out[0]->get_size() != static_cast<int>(data.size())) {
      continue;
    }

    std::vector<float> value = data;
    out[0]->set_data(value);
    constant.insert(out[0].get());
    folded.push_back(node);
    ++result.count;
  }

  for (const auto& node : folded) {
    graph.remove_node(node);
  }
  result.changed = result.count > 0;
  return result;
}

}  // namespace opt
