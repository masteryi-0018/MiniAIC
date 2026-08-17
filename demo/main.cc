// demo: 用 MiniNN 加载 MiniAIC 序列化的 .gynn 模型并推理
// （代码取自 MiniNN demo/main.cc）
#include <iostream>

#include "mininn/graph/graph.h"
#include "mininn/graph/use_kernel.h"
#include "mininn/graph/use_op.h"
#include "mininn/parser/parser.h"
#include "mininn/runtime/predictor.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_file>" << std::endl;
    return -1;
  }
  std::string filename = argv[1];
  auto graph = std::make_shared<Graph>();
  load_model(filename, graph);

  auto predictor = std::make_shared<Predictor>(graph);
  predictor->prepare();

  std::vector<std::shared_ptr<Tensor>> input_tensors =
      predictor->get_input_tensors();
  if (input_tensors.empty()) {
    std::cerr << "No input tensors found." << std::endl;
    return -1;
  }
  for (auto& tensor : input_tensors) {
    float* input = reinterpret_cast<float*>(tensor->get_buffer());
    for (int i = 0; i < tensor->get_size(); ++i) {
      input[i] = 1.0f;
    }
  }

  predictor->run();

  std::vector<std::shared_ptr<Tensor>> output_tensors =
      predictor->get_output_tensors();
  if (output_tensors.empty()) {
    std::cerr << "No output tensors found." << std::endl;
    return -1;
  }
  float* output = reinterpret_cast<float*>(output_tensors[0]->get_buffer());
  int out_size = output_tensors[0]->get_size();

  std::cout << "Output size: " << out_size << std::endl;
  int print_num = out_size < 10 ? out_size : 10;
  std::cout << "First " << print_num << " values:";
  for (int i = 0; i < print_num; ++i) {
    std::cout << " " << output[i];
  }
  std::cout << std::endl;
  return 0;
}
