#ifndef NODE_H
#define NODE_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "miniaic/ir/type.h"
#include "miniaic/ir/tensor.h"

class Node {
 public:
  Node(Op type);

  // "default" means use a default impl; "0" means you should impl it your self.
  virtual ~Node() = default;

  virtual void set_input_tensors(
      std::vector<std::shared_ptr<Tensor>>& tensors) = 0;

  virtual void set_output_tensors(
      std::vector<std::shared_ptr<Tensor>>& tensors) = 0;

  virtual void set_attributes(
      std::map<std::string, std::vector<int>>& attrs) = 0;

 private:
  Op op_type_;
};

#endif  // NODE_H