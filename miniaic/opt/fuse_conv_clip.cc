#include "miniaic/opt/fuse_conv_clip.h"

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

// 在 tensor 列表中查找下标（按指针比较）
int FindTensorIndex(const std::vector<std::shared_ptr<Tensor>>& tensors,
                    const std::shared_ptr<Tensor>& tensor) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].get() == tensor.get()) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

PassResult fuse_conv_clip(Graph& graph) {
  std::vector<std::shared_ptr<Node>>& nodes = graph.get_nodes();
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();

  // tensor 下标 -> 生产它的节点下标 / 消费它的节点下标列表
  std::map<int, int> producer;
  std::map<int, std::vector<int>> consumers;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int node_index = static_cast<int>(i);
    for (const auto& tensor : nodes[i]->get_output_tensors()) {
      producer[FindTensorIndex(tensors, tensor)] = node_index;
    }
    for (const auto& tensor : nodes[i]->get_input_tensors()) {
      consumers[FindTensorIndex(tensors, tensor)].push_back(node_index);
    }
  }

  std::set<int> graph_outputs;
  for (int tensor : graph.get_outputs()) {
    graph_outputs.insert(tensor);
  }

  int fused = 0;
  std::vector<std::shared_ptr<Node>> fused_nodes;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const std::shared_ptr<Node>& node = nodes[i];
    if (node->get_op_type() != Op::CLIP) {
      continue;
    }
    const std::vector<std::shared_ptr<Tensor>>& clip_inputs =
        node->get_input_tensors();
    if (clip_inputs.size() != 1) {
      continue;
    }
    const int src = FindTensorIndex(tensors, clip_inputs[0]);
    auto producer_it = producer.find(src);
    if (producer_it == producer.end()) {
      continue;
    }
    const int conv_index = producer_it->second;
    if (nodes[conv_index]->get_op_type() != Op::CONV) {
      continue;
    }
    // Conv 输出必须只被这个 Clip 消费，否则重连会丢数据
    auto consumers_it = consumers.find(src);
    if (consumers_it == consumers.end() ||
        consumers_it->second != std::vector<int>{static_cast<int>(i)}) {
      continue;
    }
    // Conv 输出本身是 graph 输出时不能丢
    if (graph_outputs.count(src) > 0) {
      continue;
    }
    // 需要同时有非空的 min/max 才能表达融合后的 clip
    const std::map<std::string, std::vector<int>>& clip_attrs =
        node->get_attributes();
    auto min_it = clip_attrs.find("min");
    auto max_it = clip_attrs.find("max");
    if (min_it == clip_attrs.end() || max_it == clip_attrs.end() ||
        min_it->second.empty() || max_it->second.empty()) {
      continue;
    }

    // 融合：Conv 带上 min/max，输出改为 Clip 的输出
    std::shared_ptr<Node>& conv = nodes[conv_index];
    std::map<std::string, std::vector<int>>& conv_attrs =
        conv->get_attributes();
    conv_attrs["min"] = min_it->second;
    conv_attrs["max"] = max_it->second;
    std::vector<std::shared_ptr<Tensor>> outputs = node->get_output_tensors();
    conv->set_output_tensors(outputs);

    fused_nodes.push_back(node);
    ++fused;
  }

  for (const auto& node : fused_nodes) {
    graph.remove_node(node);
  }
  return PassResult{fused > 0, fused};
}

}  // namespace opt
