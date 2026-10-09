#ifndef PASS_H
#define PASS_H

#include <functional>
#include <string>
#include <vector>

#include "miniaic/ir/graph.h"

namespace opt {

// 单个 pass 的执行结果
struct PassResult {
  bool changed = false;  // 图是否发生变化（fixpoint 迭代是否继续的依据）
  int count = 0;         // 处理的元素个数（折叠/融合/删除…），用于统计
};

// pass 函数：作用于 graph，返回执行结果
using PassFn = std::function<PassResult(Graph&)>;

// 在每个 pass 前后回调，用于 dump 中间结果（tag 形如 "01.before_shape_inference"）
using DumpCallback = std::function<void(Graph&, const std::string& tag)>;

// 图结构校验：节点引用的 tensor 都属于 graph，输入输出下标合法，
// graph 输出有生产者。pass 执行后自动调用，用于尽早发现 pass 的 bug。
bool verify(Graph& graph);

// 图优化 pass 管理器：注册、按名启用、迭代到收敛、打印统计
class PassManager {
 public:
  // fixpoint 为 true 时反复运行该 pass 直到不再变化或有上限保护
  void add(const std::string& name, PassFn fn, bool fixpoint = false);

  // 只启用给定名字的 pass（未列出的全部禁用）；含未注册的名字时返回 false
  bool set_enabled(const std::vector<std::string>& names);

  // 依次运行启用的 pass；dump_cb 非空时在每个 pass 前后回调。
  // 返回 true 表示全部 pass 执行且结构校验通过。
  bool run(Graph& graph, const DumpCallback& dump_cb = DumpCallback());

 private:
  struct PassEntry {
    std::string name;
    PassFn fn;
    bool fixpoint = false;
    bool enabled = true;
  };
  std::vector<PassEntry> passes_;
};

}  // namespace opt

#endif  // PASS_H
