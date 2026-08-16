# use flatbuffers to compile mininn.fbs

## step1: build flatc

run:
```sh
cd third_party/flatbuffers/

cmake -B build
cmake --build build
```

## step2: use flatc to gen cpp interface file

run:
```sh
../third_party/flatbuffers/build/flatc --cpp mininn.fbs
```
