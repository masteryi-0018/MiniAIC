#include "miniaic/opt/pass.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "miniaic/ir/node.h"
#include "miniaic/ir/tensor.h"

namespace opt {
namespace {

// fixpoint 迭代上限，避免有 bug 的 pass 陷入死循环
constexpr int kMaxIterations = 100;

// tag 形如 "01.before_shape_inference"，序号保证文件名顺序即执行顺序
std::string MakeTag(int seq, const char* phase, const std::string& name) {
  char index[16];
  std::snprintf(index, sizeof(index), "%02d", seq);
  return std::string(index) + "." + phase + "_" + name;
}

}  // namespace

bool verify(Graph& graph) {
  std::set<const Tensor*> known;
  std::vector<std::shared_ptr<Tensor>>& tensors = graph.get_tensors();
  for (const auto& tensor : tensors) {
    known.insert(tensor.get());
  }

  std::set<const Tensor*> produced;
  for (auto& node : graph.get_nodes()) {
    for (const auto& tensor : node->get_input_tensors()) {
      if (tensor == nullptr || known.count(tensor.get()) == 0) {
        std::cerr << "verify: unknown input tensor" << std::endl;
        return false;
      }
    }
    if (node->get_output_tensors().empty()) {
      std::cerr << "verify: node without output tensor" << std::endl;
      return false;
    }
    for (const auto& tensor : node->get_output_tensors()) {
      if (tensor == nullptr || known.count(tensor.get()) == 0) {
        std::cerr << "verify: unknown output tensor" << std::endl;
        return false;
      }
      produced.insert(tensor.get());
    }
  }

  const int num_tensors = static_cast<int>(tensors.size());
  for (int index : graph.get_inputs()) {
    if (index < 0 || index >= num_tensors) {
      std::cerr << "verify: invalid graph input index " << index << std::endl;
      return false;
    }
  }
  for (int index : graph.get_outputs()) {
    if (index < 0 || index >= num_tensors) {
      std::cerr << "verify: invalid graph output index " << index << std::endl;
      return false;
    }
    // graph 输出必须有生产者（来自某个节点或本身就是 graph 输入）
    bool from_input = false;
    for (int input : graph.get_inputs()) {
      if (input == index) {
        from_input = true;
      }
    }
    if (!from_input && produced.count(tensors[index].get()) == 0) {
      std::cerr << "verify: graph output " << index << " has no producer"
                << std::endl;
      return false;
    }
  }
  return true;
}

void PassManager::add(const std::string& name, PassFn fn, bool fixpoint) {
  passes_.push_back(PassEntry{name, std::move(fn), fixpoint, true});
}

bool PassManager::set_enabled(const std::vector<std::string>& names) {
  for (auto& pass : passes_) {
    pass.enabled = false;
  }
  bool ok = true;
  for (const std::string& name : names) {
    bool found = false;
    for (auto& pass : passes_) {
      if (pass.name == name) {
        pass.enabled = true;
        found = true;
      }
    }
    if (!found) {
      std::cerr << "Unknown pass: " << name << std::endl;
      ok = false;
    }
  }
  return ok;
}

bool PassManager::run(Graph& graph, const DumpCallback& dump_cb) {
  std::cout << "===== Graph Optimization =====" << std::endl;
  int seq = 0;
  int num_enabled = 0;
  int num_changed = 0;
  for (auto& pass : passes_) {
    if (!pass.enabled) {
      continue;
    }
    ++num_enabled;
    if (dump_cb) {
      dump_cb(graph, MakeTag(seq, "before", pass.name));
    }

    const auto start = std::chrono::steady_clock::now();
    PassResult result;
    bool any_changed = false;
    int total_count = 0;
    int iterations = 0;
    do {
      result = pass.fn(graph);
      any_changed = any_changed || result.changed;
      total_count += result.count;
      ++iterations;
    } while (pass.fixpoint && result.changed && iterations < kMaxIterations);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();

    if (dump_cb) {
      dump_cb(graph, MakeTag(seq, "after", pass.name));
    }
    ++seq;

    if (any_changed) {
      ++num_changed;
    }
    std::cout << "[pass] " << pass.name
              << ": changed=" << (any_changed ? 1 : 0)
              << " count=" << total_count << " iter=" << iterations
              << " time=" << elapsed_ms << "ms";
    if (iterations >= kMaxIterations) {
      std::cout << " (iteration limit reached)";
    }
    std::cout << std::endl;

    if (!verify(graph)) {
      std::cerr << "Graph verification failed after pass: " << pass.name
                << std::endl;
      return false;
    }
  }
  std::cout << "[pass] summary: run " << num_enabled << "/" << passes_.size()
            << " passes, " << num_changed << " changed" << std::endl;
  return true;
}

}  // namespace opt
