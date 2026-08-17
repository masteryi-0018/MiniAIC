#include "miniaic/serializer/serializer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fbs_schema/mininn_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "miniaic/ir/node.h"
#include "miniaic/ir/tensor.h"
#include "miniaic/ir/type.h"

namespace serializer {
namespace {

// 在 graph 的 tensor 列表中查找 tensor 的下标（按指针比较）
int FindTensorIndex(Graph& graph, const std::shared_ptr<Tensor>& tensor) {
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].get() == tensor.get()) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

bool save(Graph& graph, const std::string& path) {
  flatbuffers::FlatBufferBuilder builder;

  // 1. tensors：shape 为 int32 数组，data 为 float32 原始字节
  std::vector<flatbuffers::Offset<mininn_fbs::Tensor>> tensor_offsets;
  for (const auto& tensor : graph.get_tensors()) {
    const std::vector<int>& shape = tensor->get_shape();
    std::vector<int32_t> fbs_shape(shape.begin(), shape.end());
    auto shape_offset = builder.CreateVector(fbs_shape);

    // 无有效数据的 tensor（中间结果）data 写 null，与 MiniNN Python 侧一致
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_offset = 0;
    if (tensor->has_data()) {
      const uint8_t* bytes =
          reinterpret_cast<const uint8_t*>(tensor->get_buffer());
      data_offset = builder.CreateVector(
          bytes, static_cast<size_t>(tensor->get_size()) * sizeof(float));
    }

    tensor_offsets.push_back(
        mininn_fbs::CreateTensor(builder, shape_offset, data_offset));
  }
  auto tensors_offset = builder.CreateVector(tensor_offsets);

  // 2. nodes：type + attributes + inputs/outputs（tensor 下标）
  std::vector<flatbuffers::Offset<mininn_fbs::Node>> node_offsets;
  for (const auto& node : graph.get_nodes()) {
    // attributes
    std::vector<flatbuffers::Offset<mininn_fbs::Attribute>> attr_offsets;
    const std::map<std::string, std::vector<int>>& attrs =
        node->get_attributes();
    for (const auto& kv : attrs) {
      auto key_offset = builder.CreateString(kv.first);
      std::vector<int32_t> value(kv.second.begin(), kv.second.end());
      auto value_offset = builder.CreateVector(value);
      attr_offsets.push_back(
          mininn_fbs::CreateAttribute(builder, key_offset, value_offset));
    }
    auto attrs_offset = builder.CreateVector(attr_offsets);

    // inputs / outputs：shared_ptr<Tensor> -> 下标
    std::vector<int32_t> inputs;
    for (const auto& t : node->get_input_tensors()) {
      inputs.push_back(FindTensorIndex(graph, t));
    }
    std::vector<int32_t> outputs;
    for (const auto& t : node->get_output_tensors()) {
      outputs.push_back(FindTensorIndex(graph, t));
    }
    auto inputs_offset = builder.CreateVector(inputs);
    auto outputs_offset = builder.CreateVector(outputs);

    node_offsets.push_back(mininn_fbs::CreateNode(
        builder, static_cast<mininn_fbs::Op>(node->get_op_type()),
        attrs_offset, inputs_offset, outputs_offset));
  }
  auto nodes_offset = builder.CreateVector(node_offsets);

  // 3. graph inputs / outputs（tensor 下标）
  const std::vector<int>& graph_inputs = graph.get_inputs();
  const std::vector<int>& graph_outputs = graph.get_outputs();
  auto inputs_offset = builder.CreateVector(
      std::vector<int32_t>(graph_inputs.begin(), graph_inputs.end()));
  auto outputs_offset = builder.CreateVector(
      std::vector<int32_t>(graph_outputs.begin(), graph_outputs.end()));

  auto graph_offset = mininn_fbs::CreateGraph(
      builder, nodes_offset, tensors_offset, inputs_offset, outputs_offset);
  builder.Finish(graph_offset);

  // 4. 写文件
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }
  ofs.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
            static_cast<std::streamsize>(builder.GetSize()));
  ofs.close();
  std::cout << "Graph serialized to " << path << std::endl;
  return true;
}

}  // namespace serializer
