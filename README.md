# Benchmark_LLM

Code for the health check-up paper.

## 1. Building the Fine-tuning Environment

```bash
conda create -n LLM python=3.10 -y
conda activate LLM
cd ./Benchmark_LLM
pip install -r requirements.txt
```
## 2. Fine-Tuning the Seven LLMs

```bash
cd ./Benchmark_LLM/train
```

You can now run any of the seven LLM training notebooks (`.ipynb` files) to train the corresponding models.

**Note:** Before running the notebooks, you may need to modify the following configuration parameters:

- `model_name`: The Hugging Face model name or local path of the pretrained model downloaded from the Hugging Face platform.
- `dataset_path`: The path to the training dataset. Refer to `./sample/train.json` for an example of the expected data format.
- `output_dir`: The directory where the trained LoRA adapter parameters will be saved.

## 3. Merging the Base LLM and LoRA Parameters

Run the following command to merge the pretrained model parameters with the fine-tuned LoRA adapter parameters:

```bash
cd ..
python merge.py \
    --base_model path/to/llama/model \
    --tokenizer_path path/to/llama/tokenizer \
    --lora_model path/to/lora/model \
    --output_dir path/to/output/dir
```

## 4. Quantizing the Seven LLMs

First, clone the `llama.cpp` repository: [llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
# 1. Create and activate the environment
cd llama.cpp
conda create -n llama_cpp python=3.11 -y
conda activate llama_cpp

# 2. Install the required dependencies using Conda
conda install -c conda-forge \
    make \
    cmake \
    compilers \
    ninja \
    autoconf \
    automake \
    libtool \
    pkg-config \
    findutils

# 3. Build llama.cpp
# The compiled executables will be located in ./build/bin/
cmake -B build
cmake --build build --config Release

# 4. Install the conversion dependencies
pip install -r ./requirements-convert_hf_to_gguf.txt

# 5. Convert and quantize 7 LLMs
python convert_hf_to_gguf.py ./merge_pth/ --outtype q8_0 --outfile Qwen3-14B.gguf
```
- ./merge_pth/: Path to the merged model parameters generated in Section 3 (Merging the Base LLM and LoRA Parameters).
- --outtype q8_0: Specifies the 8-bit quantization format.
- --outfile: Specifies the output GGUF file name.

## 5. Inference

```bash
cd inference
conda activate llama_cpp
```
You can now run any of the seven LLM inference notebooks (`.ipynb` files).

**Note:** Before running the notebooks, you may need to modify the following configuration parameters:

- `MODEL_PATH`: Specifies the output GGUF file name.
- `INPUT_JSON `: The path to the inference dataset. Refer to `./sample/TiJian_45.json` for an example of the expected data format.
- `OUTPUT_JSON`: The directory where the inference results will be saved.
