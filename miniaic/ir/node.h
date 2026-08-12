#ifndef NODE_H
#define NODE_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "miniaic/ir/type.h"
#include "miniaic/ir/tensor.h"

// 具体类：当前所有算子（Add/Conv/...）统一用 Node 表示，
// 后续需要时再拆分为 AddNode/ConvNode 等子类。
class Node {
 public:
  explicit Node(Op type);

  virtual ~Node() = default;

  // 默认实现会保存到成员变量；子类可覆盖
  virtual void set_input_tensors(
      std::vector<std::shared_ptr<Tensor>>& tensors);

  virtual void set_output_tensors(
      std::vector<std::shared_ptr<Tensor>>& tensors);

  virtual void set_attributes(
      std::map<std::string, std::vector<int>>& attrs);

  Op get_op_type() const;

  std::vector<std::shared_ptr<Tensor>>& get_input_tensors();

  std::vector<std::shared_ptr<Tensor>>& get_output_tensors();

  std::map<std::string, std::vector<int>>& get_attributes();

 private:
  Op op_type_;
  std::vector<std::shared_ptr<Tensor>> input_tensors_;
  std::vector<std::shared_ptr<Tensor>> output_tensors_;
  std::map<std::string, std::vector<int>> attrs_;
};

#endif  // NODE_H
