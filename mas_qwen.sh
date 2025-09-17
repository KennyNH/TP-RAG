#!/bin/bash
set -e

# python3 -m agentpedia.frame.mas_v3 --filepath benchmark_few --prompt_type evolutionary_optimize --base_model qwen25-72b --multiprocess True
python3 -m agentpedia.frame.mas_v3 --filepath benchmark_popular --prompt_type evolutionary_optimize --base_model qwen25-72b --multiprocess True

# cd ../..
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_qwen_dynamic.py
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_dynamic.py --base_model qwen25-72b