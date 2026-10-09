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

// 导出前校验：
//   1. 所有 tensor 的 shape 必须已知（[-1] 是 shape_inference 留下的标记，
//      mininn 会按 size=-1 分配内存）
//   2. 节点引用的 tensor 都属于 graph，且每个节点有输出
//   3. graph 输入输出下标合法，graph 输出必须有生产者
// 校验不通过就拒绝导出，避免生成 mininn 无法正确运行的模型。
bool ValidateForExport(Graph& graph) {
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();
  std::set<const Tensor*> known;
  for (const auto& tensor : tensors) {
    known.insert(tensor.get());
    const std::vector<int>& shape = tensor->get_shape();
    if (shape.size() == 1 && shape[0] == -1) {
      std::cerr << "Validation failed: tensor has unknown shape [-1]"
                << std::endl;
      return false;
    }
  }

  std::set<const Tensor*> produced;
  for (const auto& node : graph.get_nodes()) {
    if (node->get_output_tensors().empty()) {
      std::cerr << "Validation failed: node without output tensor" << std::endl;
      return false;
    }
    for (const auto& tensor : node->get_input_tensors()) {
      if (tensor == nullptr || known.count(tensor.get()) == 0) {
        std::cerr << "Validation failed: node input not in graph" << std::endl;
        return false;
      }
    }
    for (const auto& tensor : node->get_output_tensors()) {
      if (tensor == nullptr || known.count(tensor.get()) == 0) {
        std::cerr << "Validation failed: node output not in graph"
                  << std::endl;
        return false;
      }
      produced.insert(tensor.get());
    }
  }

  const int num_tensors = static_cast<int>(tensors.size());
  std::set<int> graph_inputs;
  for (int index : graph.get_inputs()) {
    if (index < 0 || index >= num_tensors) {
      std::cerr << "Validation failed: invalid graph input index " << index
                << std::endl;
      return false;
    }
    graph_inputs.insert(index);
  }
  for (int index : graph.get_outputs()) {
    if (index < 0 || index >= num_tensors) {
      std::cerr << "Validation failed: invalid graph output index " << index
                << std::endl;
      return false;
    }
    if (graph_inputs.count(index) == 0 &&
        produced.count(tensors[index].get()) == 0) {
      std::cerr << "Validation failed: graph output " << index
                << " has no producer" << std::endl;
      return false;
    }
  }
  return true;
}

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

const char* OpToString(Op op) {
  switch (op) {
    case Op::ADD:
      return "Add";
    case Op::CONV:
      return "Conv";
    case Op::CLIP:
      return "Clip";
    case Op::SHAPE:
      return "Shape";
    case Op::GATHER:
      return "Gather";
    case Op::UNSQUEEZE:
      return "Unsqueeze";
    case Op::CONCAT:
      return "Concat";
    case Op::GLOBALAVERAGEPOOL:
      return "GlobalAveragePool";
    case Op::RESHAPE:
      return "Reshape";
    case Op::GEMM:
      return "Gemm";
    case Op::CONSTANT:
      return "Constant";
  }
  return "Unknown";
}

}  // namespace

bool save(Graph& graph, const std::string& path) {
  if (!ValidateForExport(graph)) {
    std::cerr << "Refuse to serialize " << path << std::endl;
    return false;
  }

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

// 输出 onnx ModelProto 的 camelCase JSON（Netron 对 .json 的 onnx 变体解析方式），
// 不含权重数据，所有 tensor 带 shape（来自 shape inference）
bool save_json(Graph& graph, const std::string& path) {
  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return false;
  }

  // tensor 名字 = 下标字符串
  auto tensor_name = [](int index) { return std::to_string(index); };

  // 输出 value info：{"name": .., "type": {"tensorType": {"elemType": 1, "shape": {"dim": [...]}}}}
  auto print_value_info = [&](const std::vector<int>& tensors) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      ofs << (i > 0 ? ", " : "")
          << "{\"name\": \"" << tensor_name(tensors[i]) << "\", ";
      ofs << "\"type\": {\"tensorType\": {\"elemType\": 1, ";
      const std::vector<int>& shape =
          graph.get_tensors()[tensors[i]]->get_shape();
      ofs << "\"shape\": {\"dim\": [";
      for (size_t j = 0; j < shape.size(); ++j) {
        if (j > 0) {
          ofs << ", ";
        }
        ofs << "{\"dimValue\": " << shape[j] << "}";
      }
      // ] 结束 dim 数组，4 个 } 依次闭合 shape / tensorType / type / value_info 对象
      ofs << "]}}}}";
    }
  };

  ofs << "{\n";
  ofs << "  \"irVersion\": 6,\n";
  ofs << "  \"producerName\": \"miniaic\",\n";
  ofs << "  \"opsetImport\": [{\"domain\": \"\", \"version\": 10}],\n";
  ofs << "  \"graph\": {\n";

  // 1. nodes
  ofs << "    \"node\": [\n";
  std::vector<std::shared_ptr<Node>>& nodes = graph.get_nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    ofs << "      {\"opType\": \"" << OpToString(node->get_op_type())
        << "\", \"name\": \"" << i << "\", ";

    // inputs：tensor 名（下标字符串）
    ofs << "\"input\": [";
    const auto& inputs = node->get_input_tensors();
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (j > 0) {
        ofs << ", ";
      }
      ofs << "\"" << tensor_name(FindTensorIndex(graph, inputs[j])) << "\"";
    }
    ofs << "], ";

    // outputs
    ofs << "\"output\": [";
    const auto& outputs = node->get_output_tensors();
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (j > 0) {
        ofs << ", ";
      }
      ofs << "\"" << tensor_name(FindTensorIndex(graph, outputs[j])) << "\"";
    }
    ofs << "], ";

    // attributes：单元素 -> i，多元素 -> ints（IR attrs 只存 int）
    ofs << "\"attribute\": [";
    const std::map<std::string, std::vector<int>>& attrs =
        node->get_attributes();
    size_t attr_count = 0;
    for (const auto& kv : attrs) {
      if (attr_count++ > 0) {
        ofs << ", ";
      }
      ofs << "{\"name\": \"" << kv.first << "\", ";
      // 必须显式给 type：Netron 的 AttributeProto.f 默认就是 0，
      // 若省略 type 会把单元素 int 属性误判成 FLOAT 并显示成 0
      if (kv.second.size() == 1) {
        ofs << "\"type\": \"INT\", \"i\": " << kv.second[0];
      } else {
        ofs << "\"type\": \"INTS\", \"ints\": [";
        for (size_t j = 0; j < kv.second.size(); ++j) {
          if (j > 0) {
            ofs << ", ";
          }
          ofs << kv.second[j];
        }
        ofs << "]";
      }
      ofs << "}";
    }
    ofs << "]}";
    if (i + 1 < nodes.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "    ],\n";

  // 2. graph inputs / outputs / valueInfo（全部带 shape）
  ofs << "    \"input\": [";
  print_value_info(graph.get_inputs());
  ofs << "],\n";

  ofs << "    \"output\": [";
  print_value_info(graph.get_outputs());
  ofs << "],\n";

  // valueInfo：不在 input/output 里的所有 tensor（中间结果）
  ofs << "    \"valueInfo\": [";
  const std::vector<int>& graph_inputs = graph.get_inputs();
  const std::vector<int>& graph_outputs = graph.get_outputs();
  std::vector<int> intermediates;
  for (int i = 0; i < static_cast<int>(graph.get_tensors().size()); ++i) {
    const bool is_io =
        std::find(graph_inputs.begin(), graph_inputs.end(), i) !=
            graph_inputs.end() ||
        std::find(graph_outputs.begin(), graph_outputs.end(), i) !=
            graph_outputs.end();
    if (!is_io) {
      intermediates.push_back(i);
    }
  }
  print_value_info(intermediates);
  ofs << "]\n";

  ofs << "  }\n";
  ofs << "}\n";
  ofs.close();
  std::cout << "Graph dumped to " << path << " (ONNX JSON)" << std::endl;
  return true;
}

}  // namespace serializer
