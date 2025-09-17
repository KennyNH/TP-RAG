# TP-RAG: Benchmarking Retrieval-Augmented Large Language Model Agents for Spatiotemporal-Aware Travel Planning
This is the repo for the paper: TP-RAG: Benchmarking Retrieval-Augmented Large Language Model Agents for Spatiotemporal-Aware Travel Planning, which has been accepted by EMNLP'2025 Main.

# Data
The dataset is available in [Google Drive](https://drive.google.com/file/d/11b7-W3Q6bpDWLrkr2wJ7ZhcDItXcpRQo/view?usp=sharing).

# Dependencies
python=3.9.18, vllm=0.7.1

# Note
1. For LLM prompting, your should set your own base url and api keys in "agentpedia/config/basic_config.yaml". We apply vllm for the inference of LLaMA, Qwen, and DeepSeek models.
2. You should download the data from Google Drive and place them in "construct_data" and "construct_data_baseline" folders (created by yourself).
3. You may need replace some dict keys of "construct_data" in "agentpedia/frame/baseline.py" and "agentpedia/frame/mas_v3.py", which are inconsistent with those in our released dataset in Google Drive. You can refer to "data_process.py" for the details about dict key replacement.

# Benchmark

```
bash method_qwen.sh
bash eval_qwen.sh
```

# EvoRAG
```
bash mas_qwen.sh
bash eval_mas_qwen.sh
```
