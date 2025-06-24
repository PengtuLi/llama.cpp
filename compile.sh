# !bin bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Release -j12
