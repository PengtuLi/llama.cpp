# !bin bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_DEBUG=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release  --target llama-simple -j12
