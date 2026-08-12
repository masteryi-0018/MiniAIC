#include "miniaic/ir/node.h"

Node::Node(Op type) : op_type_(type) {}

Op Node::get_op_type() const { return op_type_; }

void Node::set_input_tensors(
    std::vector<std::shared_ptr<Tensor>>& tensors) {
  input_tensors_ = tensors;
}

void Node::set_output_tensors(
    std::vector<std::shared_ptr<Tensor>>& tensors) {
  output_tensors_ = tensors;
}

void Node::set_attributes(
    std::map<std::string, std::vector<int>>& attrs) {
  attrs_ = attrs;
}

std::vector<std::shared_ptr<Tensor>>& Node::get_input_tensors() {
  return input_tensors_;
}

std::vector<std::shared_ptr<Tensor>>& Node::get_output_tensors() {
  return output_tensors_;
}

std::map<std::string, std::vector<int>>& Node::get_attributes() {
  return attrs_;
}
