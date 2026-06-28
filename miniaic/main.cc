#include <iostream>
#include <fstream>

#include "miniaic/onnx/onnx.pb.h"

int main() {
    // 1. 读取整个 .onnx 文件到 string
    std::ifstream ifs("models/add_model.onnx", std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open models/add_model.onnx" << std::endl;
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

    // 3. 打印一些基本信息（示例）
    std::cout << "IR version: " << model.ir_version() << std::endl;
    std::cout << "Producer: " << model.producer_name() << std::endl;
    std::cout << "Graph name: " << model.graph().name() << std::endl;
    std::cout << "Number of nodes: " << model.graph().node_size() << std::endl;

    // 遍历所有节点
    for (const auto& node : model.graph().node()) {
        std::cout << "OpType: " << node.op_type()
                  << ", Name: " << node.name() << std::endl;
    }

    return 0;
}