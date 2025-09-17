#!/bin/bash
set -e

# base_model="gpt-4o"
# base_model="deepseek-chat"
base_model="qwen25-72b"
# base_model="llama33-70b"

# baselines

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_cot_objective --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_refine_objective --base_model $base_model --multiprocess True

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_all --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_half --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_one --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_selective_half --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_selective_one --base_model $base_model --multiprocess True
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_abstractive --base_model $base_model --multiprocess True

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type reference_denoise --base_model $base_model --multiprocess True --num_threads 25
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type reference_best --base_model $base_model --multiprocess True --num_threads 25
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type reference --base_model $base_model --multiprocess True --num_threads 25




# in-depth analysis (sensitivity analysis)

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N7 --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N6 --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N5 --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N3 --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N2 --base_model $base_model --multiprocess True --num_threads 20

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_all_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_half_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_one_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N7_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N6_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N5_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N3_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_N2_clean --base_model $base_model --multiprocess True --num_threads 20

python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_selective_half_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_selective_one_clean --base_model $base_model --multiprocess True --num_threads 20
python3 -m agentpedia.frame.baseline --filepath benchmark_popular --prompt_type given_direct_objective_retrieval_abstractive_clean --base_model $base_model --multiprocess True --num_threads 20

# cd ../..
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_qwen_dynamic.py
# python /hpc2hdd/home/hni017/Workplace/LLMFactory/occupy_dynamic.py --base_model $base_model