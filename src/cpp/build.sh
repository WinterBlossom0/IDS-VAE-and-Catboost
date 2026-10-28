#!/usr/bin/env bash
# Builds the compiled timing program against the ONNX Runtime 1.31.0 release and nlohmann/json 3.11.3 (both downloaded into
# third_party/ and checked by SHA-256). -ffp-contract=off keeps every floating-point operation separate, as in Python and
# NumPy, so that the C++ stream states equal the Python states.
set -euo pipefail
cd "$(dirname "$0")/../.."
ORT_VERSION=1.31.0
ORT=third_party/onnxruntime-linux-x64-$ORT_VERSION
mkdir -p third_party/nlohmann build
if [ ! -d "$ORT" ]; then
  curl -sSL -o third_party/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"
  echo "cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57  third_party/ort.tgz" | sha256sum -c -
  tar xzf third_party/ort.tgz -C third_party
fi
if [ ! -f third_party/nlohmann/json.hpp ]; then
  curl -sSL -o third_party/nlohmann/json.hpp https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
fi
echo "9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6  third_party/nlohmann/json.hpp" | sha256sum -c --quiet -
g++ -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra -I "$ORT/include" -I third_party src/cpp/timing.cpp \
  -L "$ORT/lib" -lonnxruntime -Wl,-rpath,"$PWD/$ORT/lib" -o build/timing
