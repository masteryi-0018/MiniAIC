#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "miniaic/ir/graph.h"
#include "miniaic/onnx/onnx.pb.h"
#include "miniaic/opt/shape_inference.h"
#include "miniaic/serializer/serializer.h"

namespace {

// 将 ONNX 的 op_type 字符串映射为内部 IR 的 Op 枚举。
bool GetOpType(const std::string& op_type, Op* op) {
  static const std::map<std::string, Op> kOpMap = {
      {"Add", Op::ADD},
      {"Conv", Op::CONV},
      {"Clip", Op::CLIP},
      {"Shape", Op::SHAPE},
      {"Gather", Op::GATHER},
      {"Unsqueeze", Op::UNSQUEEZE},
      {"Concat", Op::CONCAT},
      {"GlobalAveragePool", Op::GLOBALAVERAGEPOOL},
      {"Reshape", Op::RESHAPE},
      {"Gemm", Op::GEMM},
      {"Constant", Op::CONSTANT},
  };
  auto it = kOpMap.find(op_type);
  if (it == kOpMap.end()) {
    return false;
  }
  *op = it->second;
  return true;
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

// 从 ValueInfoProto 中提取 shape；动态维度（dim_param）或缺失信息时返回空 shape。
std::vector<int> GetShapeFromValueInfo(const onnx::ValueInfoProto& info) {
  std::vector<int> shape;
  if (!info.has_type()) {
    return shape;
  }
  const auto& type = info.type();
  if (!type.has_tensor_type()) {
    return shape;
  }
  const auto& tensor_type = type.tensor_type();
  if (!tensor_type.has_shape()) {
    return shape;
  }
  const auto& tensor_shape = tensor_type.shape();
  for (int i = 0; i < tensor_shape.dim_size(); ++i) {
    const auto& dim = tensor_shape.dim(i);
    if (dim.has_dim_value()) {
      shape.push_back(static_cast<int>(dim.dim_value()));
    } else {
      // 动态维度（只有 dim_param）：按 1 处理（与 MiniNN Python 侧一致）
      shape.push_back(1);
    }
  }
  return shape;
}

// 将 ONNX TensorProto 中的数据转为 float 数组。
// 内部 IR 的 Tensor 目前统一按 float 存储。
std::vector<float> GetTensorData(const onnx::TensorProto& tensor) {
  std::vector<float> data;
  switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT: {
      if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        const float* p = reinterpret_cast<const float*>(raw.data());
        data.assign(p, p + raw.size() / sizeof(float));
      } else {
        for (int i = 0; i < tensor.float_data_size(); ++i) {
          data.push_back(tensor.float_data(i));
        }
      }
      break;
    }
    case onnx::TensorProto::INT32: {
      if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
        for (size_t i = 0; i < raw.size() / sizeof(int32_t); ++i) {
          data.push_back(static_cast<float>(p[i]));
        }
      } else {
        for (int i = 0; i < tensor.int32_data_size(); ++i) {
          data.push_back(static_cast<float>(tensor.int32_data(i)));
        }
      }
      break;
    }
    case onnx::TensorProto::INT64: {
      if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        const int64_t* p = reinterpret_cast<const int64_t*>(raw.data());
        for (size_t i = 0; i < raw.size() / sizeof(int64_t); ++i) {
          data.push_back(static_cast<float>(p[i]));
        }
      } else {
        for (int i = 0; i < tensor.int64_data_size(); ++i) {
          data.push_back(static_cast<float>(tensor.int64_data(i)));
        }
      }
      break;
    }
    default:
      std::cerr << "Unsupported tensor data type: " << tensor.data_type()
                << std::endl;
      break;
  }
  return data;
}

// 在 graph 的 tensor 列表中找到某个 tensor 的下标（按指针比较）。
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

int main(int argc, char* argv[]) {
  // 1. 读取整个 .onnx 文件到 string
  std::string model_path = argc > 1 ? argv[1] : "models/add_model.onnx";
  std::ifstream ifs(model_path, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open " << model_path << std::endl;
    return -1;
  }
  std::string model_str((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
  ifs.close();

  // 2. 用 Protobuf 反序列化
  onnx::ModelProto model;
  if (!model.ParseFromString(model_str)) {
    std::cerr << "Failed to parse ONNX model" << std::endl;
    return -1;
  }

  // 打印一些基本信息（示例）
  std::cout << "IR version: " << model.ir_version() << std::endl;
  std::cout << "Producer: " << model.producer_name() << std::endl;
  std::cout << "Graph name: " << model.graph().name() << std::endl;
  std::cout << "Number of nodes: " << model.graph().node_size() << std::endl;

  // 3. 将 ONNX graph 转换为内部 IR Graph
  const onnx::GraphProto& graph_proto = model.graph();
  Graph graph;

  // name -> graph 中 tensor 的下标
  std::map<std::string, int> tensor_map;

  // 收集所有已知的 shape（graph 输入/输出 + value_info），用于中间 tensor 的 shape 推断
  std::map<std::string, std::vector<int>> known_shapes;
  for (int i = 0; i < graph_proto.input_size(); ++i) {
    const auto& info = graph_proto.input(i);
    known_shapes[info.name()] = GetShapeFromValueInfo(info);
  }
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    const auto& info = graph_proto.output(i);
    known_shapes[info.name()] = GetShapeFromValueInfo(info);
  }
  for (int i = 0; i < graph_proto.value_info_size(); ++i) {
    const auto& info = graph_proto.value_info(i);
    known_shapes[info.name()] = GetShapeFromValueInfo(info);
  }

  // 3.1 注册 graph 输入 tensor
  for (int i = 0; i < graph_proto.input_size(); ++i) {
    const auto& input = graph_proto.input(i);
    if (tensor_map.count(input.name()) > 0) {
      continue;
    }
    std::vector<int> shape = GetShapeFromValueInfo(input);
    graph.add_tensor(shape, nullptr);
    tensor_map[input.name()] = static_cast<int>(graph.get_tensors().size()) - 1;
  }

  // 3.2 注册 initializer（常量权重）tensor
  for (int i = 0; i < graph_proto.initializer_size(); ++i) {
    const auto& init = graph_proto.initializer(i);
    if (tensor_map.count(init.name()) > 0) {
      continue;
    }
    std::vector<int> shape;
    for (int j = 0; j < init.dims_size(); ++j) {
      shape.push_back(static_cast<int>(init.dims(j)));
    }
    std::vector<float> data = GetTensorData(init);
    graph.add_tensor(shape, data.data());
    tensor_map[init.name()] = static_cast<int>(graph.get_tensors().size()) - 1;
  }

  // 3.3 为每个 node 的输出注册占位 tensor，
  //     这样后续 add_node 时可以通过下标找到对应的 output tensor
  for (int i = 0; i < graph_proto.node_size(); ++i) {
    const auto& node_proto = graph_proto.node(i);
    for (int j = 0; j < node_proto.output_size(); ++j) {
      const std::string& out_name = node_proto.output(j);
      if (out_name.empty() || tensor_map.count(out_name) > 0) {
        continue;
      }
      auto it = known_shapes.find(out_name);
      // 未知 shape 用 {-1} 标记（空 shape 表示标量），由 shape_inference 填充
      std::vector<int> shape = {-1};
      if (it != known_shapes.end() && !it->second.empty()) {
        shape = it->second;
      }
      // Constant 节点的 value 属性（TENSOR）直接注册为输出 tensor（带 shape 和数据），
      // 供 shape_inference 中的 Gather/Concat 等算子读取
      const void* data = nullptr;
      std::vector<float> const_data;
      if (node_proto.op_type() == "Constant") {
        for (int k = 0; k < node_proto.attribute_size(); ++k) {
          const auto& attr = node_proto.attribute(k);
          if (attr.type() == onnx::AttributeProto::TENSOR) {
            const auto& value = attr.t();
            shape.clear();
            for (int d = 0; d < value.dims_size(); ++d) {
              shape.push_back(static_cast<int>(value.dims(d)));
            }
            const_data = GetTensorData(value);
            if (!const_data.empty()) {
              data = const_data.data();
            }
            break;
          }
        }
      }
      graph.add_tensor(shape, data);
      tensor_map[out_name] = static_cast<int>(graph.get_tensors().size()) - 1;
    }
  }

  // 3.4 确保 graph 输出 tensor 已注册（通常已是某个 node 的输出）
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    const auto& output = graph_proto.output(i);
    if (tensor_map.count(output.name()) == 0) {
      std::vector<int> shape = GetShapeFromValueInfo(output);
      graph.add_tensor(shape, nullptr);
      tensor_map[output.name()] =
          static_cast<int>(graph.get_tensors().size()) - 1;
    }
  }

  // 3.5 设置 graph 的输入/输出 tensor 下标
  std::vector<int> graph_inputs;
  for (int i = 0; i < graph_proto.input_size(); ++i) {
    const std::string& name = graph_proto.input(i).name();
    auto it = tensor_map.find(name);
    if (it != tensor_map.end()) {
      graph_inputs.push_back(it->second);
    }
  }
  graph.set_inputs(graph_inputs);

  std::vector<int> graph_outputs;
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    const std::string& name = graph_proto.output(i).name();
    auto it = tensor_map.find(name);
    if (it != tensor_map.end()) {
      graph_outputs.push_back(it->second);
    }
  }
  graph.set_outputs(graph_outputs);

  // 3.6 遍历所有节点，转换为内部 IR 节点
  for (int i = 0; i < graph_proto.node_size(); ++i) {
    const auto& node_proto = graph_proto.node(i);

    Op op;
    if (!GetOpType(node_proto.op_type(), &op)) {
      std::cerr << "Unsupported op type: " << node_proto.op_type() << std::endl;
      return -1;
    }

    std::vector<int> inputs;
    for (int j = 0; j < node_proto.input_size(); ++j) {
      const std::string& in_name = node_proto.input(j);
      if (in_name.empty()) {
        continue;  // 可选输入未提供
      }
      auto it = tensor_map.find(in_name);
      if (it == tensor_map.end()) {
        std::cerr << "Input tensor not found: " << in_name << std::endl;
        return -1;
      }
      inputs.push_back(it->second);
    }

    std::vector<int> outputs;
    for (int j = 0; j < node_proto.output_size(); ++j) {
      const std::string& out_name = node_proto.output(j);
      auto it = tensor_map.find(out_name);
      if (it == tensor_map.end()) {
        std::cerr << "Output tensor not found: " << out_name << std::endl;
        return -1;
      }
      outputs.push_back(it->second);
    }

    // 属性：内部 IR 目前只支持 int 类型的属性（如 kernel_shape/strides/pads 等），
    // 不支持（如 tensor/string 等）的属性暂时跳过并告警
    std::map<std::string, std::vector<int>> attrs;
    for (int j = 0; j < node_proto.attribute_size(); ++j) {
      const auto& attr = node_proto.attribute(j);
      // Constant 的 value（TENSOR）已在注册 tensor 时处理，这里跳过
      if (node_proto.op_type() == "Constant" &&
          attr.type() == onnx::AttributeProto::TENSOR) {
        continue;
      }
      std::vector<int> values;
      switch (attr.type()) {
        case onnx::AttributeProto::INT:
          values.push_back(static_cast<int>(attr.i()));
          break;
        case onnx::AttributeProto::INTS:
          for (int k = 0; k < attr.ints_size(); ++k) {
            values.push_back(static_cast<int>(attr.ints(k)));
          }
          break;
        case onnx::AttributeProto::FLOAT:
          // 当前 IR 属性只支持 int，float 先截断存储（会丢失精度）
          values.push_back(static_cast<int>(attr.f()));
          break;
        case onnx::AttributeProto::FLOATS:
          for (int k = 0; k < attr.floats_size(); ++k) {
            values.push_back(static_cast<int>(attr.floats(k)));
          }
          break;
        default:
          std::cerr << "Unsupported attribute type: " << attr.type()
                    << " for attr: " << attr.name() << std::endl;
          break;
      }
      attrs[attr.name()] = values;
    }

    std::shared_ptr<Node> node = std::make_shared<Node>(op);
    graph.add_node(node, inputs, outputs, attrs);
  }

  // 4. 形状推断（图优化的第一步）
  opt::shape_inference(graph);

  // 5. 序列化为 .gynn flatbuffer（与 MiniNN 格式一致）
  std::string gym_path = model_path;
  if (gym_path.size() > 5 &&
      gym_path.substr(gym_path.size() - 5) == ".onnx") {
    gym_path = gym_path.substr(0, gym_path.size() - 5) + ".gynn";
  } else {
    gym_path += ".gynn";
  }
  serializer::save(graph, gym_path);

  // 6. 打印内部 IR Graph 的信息
  std::cout << "\n===== Internal IR Graph =====" << std::endl;

  std::cout << "Graph inputs (tensor indices): ";
  for (int idx : graph.get_inputs()) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;

  std::cout << "Graph outputs (tensor indices): ";
  for (int idx : graph.get_outputs()) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;

  std::cout << "Tensors (" << graph.get_tensors().size() << "):" << std::endl;
  for (size_t i = 0; i < graph.get_tensors().size(); ++i) {
    const auto& tensor = graph.get_tensors()[i];
    const std::vector<int>& shape = tensor->get_shape();
    std::cout << "  [" << i << "] shape=[";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j + 1 < shape.size()) {
        std::cout << ", ";
      }
    }
    std::cout << "] size=" << tensor->get_size() << std::endl;
  }

  std::cout << "Nodes (" << graph.get_nodes().size() << "):" << std::endl;
  for (size_t i = 0; i < graph.get_nodes().size(); ++i) {
    const auto& node = graph.get_nodes()[i];
    std::cout << "  [" << i << "] op=" << OpToString(node->get_op_type())
              << " (" << static_cast<int>(node->get_op_type()) << ")"
              << std::endl;

    std::cout << "    inputs:";
    for (const auto& tensor : node->get_input_tensors()) {
      std::cout << " " << FindTensorIndex(graph, tensor);
    }
    std::cout << std::endl;

    std::cout << "    outputs:";
    for (const auto& tensor : node->get_output_tensors()) {
      std::cout << " " << FindTensorIndex(graph, tensor);
    }
    std::cout << std::endl;

    const auto& attrs = node->get_attributes();
    std::cout << "    attrs:";
    if (attrs.empty()) {
      std::cout << " (none)";
    } else {
      for (const auto& kv : attrs) {
        std::cout << " " << kv.first << "=[";
        const auto& values = kv.second;
        for (size_t j = 0; j < values.size(); ++j) {
          std::cout << values[j];
          if (j + 1 < values.size()) {
            std::cout << ", ";
          }
        }
        std::cout << "]";
      }
    }
    std::cout << std::endl;
  }

  return 0;
}
