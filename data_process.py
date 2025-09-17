import os
import pickle
import json
from tqdm import tqdm

def transform(folder_path):
    
    os.makedirs(f"{folder_path}_json", exist_ok=True)
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(folder_path, filename)
            json_filename = filename[:-4] + '.json'
            json_path = os.path.join(f"{folder_path}_json", json_filename)
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict):
                print(f"{filename} 不是一个dict，跳过。")
                raise ValueError
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

# transform("/hpc2hdd/home/hni017/Workplace/LLMFactory/projects/main-250206/construct_data")
# transform("/hpc2hdd/home/hni017/Workplace/LLMFactory/projects/main-250206/construct_data_baseline")

def simplify(gt_path, base_path):
    
    open_path = "construct_data_open"
    os.makedirs(open_path, exist_ok=True)
    open_dict = {}
    for filename in tqdm(os.listdir(base_path)):
        query = filename.strip("-data_construct.json")
        with open(os.path.join(base_path, filename), 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        with open(os.path.join(gt_path, filename), 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        if len(base_data["plan_extract_result_list"]) < 8:
            continue
        open_dict[query] = {
            "query": query, 
            "poi_list": base_data["poi_list_no_cheat"], 
            "poi_list_sorted_by_popularity": gt_data["poi_extract_result_improve"], 
            "trajectories": base_data["plan_extract_result_list"], 
            "trajectories_clean": base_data["plan_extract_result_list_clean"],
        }
    print("# Queries: ", len(open_dict))
    with open(os.path.join(open_path, f"data.json"), 'w', encoding='utf-8') as f:
        json.dump(open_dict, f, ensure_ascii=False, indent=4)
        
simplify(gt_path="/hpc2hdd/home/hni017/Workplace/LLMFactory/projects/main-250206/construct_data_json", 
         base_path="/hpc2hdd/home/hni017/Workplace/LLMFactory/projects/main-250206/construct_data_baseline_json")