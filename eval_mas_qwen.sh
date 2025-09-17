#!/bin/bash
set -e

# python3 -m agentpedia.frame.eval --filepath benchmark_few --base_model qwen25-72b --llm_eval qwen25-72b --include_mas True --multiprocess True
python3 -m agentpedia.frame.eval --filepath benchmark_popular --base_model qwen25-72b --llm_eval qwen25-72b --include_mas True --multiprocess True

# cd ../..
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_qwen_dynamic.py
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_dynamic.py --base_model qwen25-72b