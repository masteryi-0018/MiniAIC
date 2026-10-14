# MiniAIC

Build a AI compiler from scratch. It only does **graph optimization**, no kernel
generation: takes ONNX, writes a flatbuffer (`.gynn`) for
[MiniNN](https://github.com/masteryi-0018/MiniNN) to run.

## Build

```sh
cmake -B build -S .
cmake --build build
```

## Usage

```sh
# optimize and export: models/xxx.onnx -> models/xxx.gynn (plus a .json for Netron)
./build/miniaic/frontend/main models/mobilenetv2-10.onnx

# skip the optimizations (keeps only the required shape inference) to see what they do
./build/miniaic/frontend/main models/mobilenetv2-10.onnx --no-opt

# run it with MiniNN (build MiniNN's lib first, see demo/CMakeLists.txt for the paths)
./build/demo/demo models/mobilenetv2-10.gynn
```

## Optimization passes (`miniaic/opt/`)

Run in order, each one depends on the result of the previous:

| pass | what it does |
| --- | --- |
| `shape_inference` | infers the shape of every tensor, and evaluates the constants on shape chains |
| `constant_folding` | evaluates and removes nodes whose result is already known at compile time |
| `fuse_conv_clip` | fuses `Conv + Clip` into a single `Conv` (the clip bounds become attributes) |
| `dce` | removes unreachable nodes, reclaims dead tensors and remaps tensor indices |

Before serializing, the graph is validated (unknown shapes, tensor ownership, graph
outputs must have a producer), and the `.gynn` is not written if the check fails.

## Layout

| directory | contents |
| --- | --- |
| `miniaic/frontend/` | read ONNX, build the IR, run the passes, export |
| `miniaic/ir/` | internal IR: graph / node / tensor / type |
| `miniaic/opt/` | graph optimization passes |
| `miniaic/serializer/` | IR -> `.gynn`, Netron JSON |
| `proto/` | how to generate the ONNX protobuf bindings |
| `fbs_schema/` | how to generate the `.gynn` flatbuffers schema |
| `demo/` | run inference with MiniNN |

More background: [docs/README.md](docs/README.md).
