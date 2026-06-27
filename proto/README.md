# use protobuf to compile onnx.proto

## step1: build protoc

run:
```sh
cd third_party/protobuf/

cmake -B build
cmake --build build
```

## step2: use protoc to gen cpp interface files

run:
```sh
../third_party/protobuf/build/protoc -I ../third_party/onnx --cpp_out=. onnx/onnx.proto
```