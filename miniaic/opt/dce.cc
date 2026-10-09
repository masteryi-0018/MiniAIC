#include "miniaic/opt/dce.h"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "miniaic/ir/node.h"
#include "miniaic/ir/tensor.h"
#include "miniaic/ir/type.h"

namespace opt {
namespace {

int NumElements(const std::vector<int>& shape) {
  int size = 1;
  for (int dim : shape) {
    size *= dim;
  }
  return size;
}

}  // namespace

bool dce(Graph& graph) {
  std::vector<std::shared_ptr<Node>>& nodes = graph.get_nodes();
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();

  // 生产者：tensor -> 产生它的节点下标
  std::map<const Tensor*, int> producer;
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (const auto& tensor : nodes[i]->get_output_tensors()) {
      producer[tensor.get()] = static_cast<int>(i);
    }
  }

  // 活跃性传播：从 graph 输出出发，沿生产者反向标记
  std::set<int> live_nodes;
  std::set<const Tensor*> live_tensors;
  std::vector<const Tensor*> pending;
  for (int index : graph.get_outputs()) {
    const Tensor* tensor = tensors[index].get();
    live_tensors.insert(tensor);
    pending.push_back(tensor);
  }
  while (!pending.empty()) {
    const Tensor* tensor = pending.back();
    pending.pop_back();
    auto it = producer.find(tensor);
    if (it == producer.end()) {
      continue;  // graph 输入、initializer 等没有生产者
    }
    const int node_index = it->second;
    if (live_nodes.count(node_index) > 0) {
      continue;
    }
    live_nodes.insert(node_index);
    for (const auto& in : nodes[node_index]->get_input_tensors()) {
      if (live_tensors.insert(in.get()).second) {
        pending.push_back(in.get());
      }
    }
    for (const auto& out : nodes[node_index]->get_output_tensors()) {
      live_tensors.insert(out.get());
    }
  }

  // 需要保留的张量：graph 输入输出 + 活跃节点用到的张量
  std::set<const Tensor*> keep = live_tensors;
  for (int index : graph.get_inputs()) {
    keep.insert(tensors[index].get());
  }

  // 删除不活跃节点
  std::vector<std::shared_ptr<Node>> dead_nodes;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (live_nodes.count(static_cast<int>(i)) == 0) {
      dead_nodes.push_back(nodes[i]);
    }
  }
  for (const auto& node : dead_nodes) {
    graph.remove_node(node);
  }

  // 回收死张量并重排下标，同时统计释放的 buffer 大小
  std::vector<std::shared_ptr<Tensor>> kept;
  std::vector<int> old_to_new(tensors.size(), -1);
  size_t freed_bytes = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (keep.count(tensors[i].get()) > 0) {
      old_to_new[i] = static_cast<int>(kept.size());
      kept.push_back(tensors[i]);
    } else {
      freed_bytes += static_cast<size_t>(NumElements(tensors[i]->get_shape())) *
                     sizeof(float);
    }
  }
  const int removed_tensors = static_cast<int>(tensors.size() - kept.size());
  const int removed_nodes = static_cast<int>(dead_nodes.size());

  std::vector<int> new_inputs;
  for (int index : graph.get_inputs()) {
    new_inputs.push_back(old_to_new[index]);
  }
  std::vector<int> new_outputs;
  for (int index : graph.get_outputs()) {
    new_outputs.push_back(old_to_new[index]);
  }
  tensors = kept;  // tensors 是 graph 内部 vector 的引用
  graph.set_inputs(new_inputs);
  graph.set_outputs(new_outputs);

  if (removed_nodes + removed_tensors > 0) {
    std::cout << "DCE: removed " << removed_nodes << " node(s), "
              << removed_tensors << " tensor(s), freed " << freed_bytes / 1024
              << " KB" << std::endl;
  }
  return removed_nodes + removed_tensors > 0;
}

}  // namespace opt
