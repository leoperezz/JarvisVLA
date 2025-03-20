import argparse
from pathlib import Path
import shutil
import os
import safetensors
from rich import print
import json
from jarvisvla.utils.file_utils import load_json_file,dump_json_file
from jarvisvla import QWEN_SPECIAL_TOKENS

def apply_full_model(args):
    base_model_path = args.base_model_path.lower().replace('-','_')
    final_model_path = args.final_model_path
    if args.enable_processor:
        
        if  "qwen2_vl" in base_model_path:
            from transformers import Qwen2VLProcessor
            processor = Qwen2VLProcessor.from_pretrained(args.base_model_path)
            with open(QWEN_SPECIAL_TOKENS, "r") as file:
                special_token = json.load(file)
            processor.tokenizer.add_special_tokens({"additional_special_tokens":special_token})
            processor.save_pretrained(final_model_path)
        else:
            raise ValueError(f"model not find: {base_model_path}")
        

        config_file = Path(args.base_model_path) / "preprocessor_config.json"
        target_path = Path(final_model_path) / "preprocessor_config.json"
        
        if config_file.exists():
            shutil.copy(str(config_file), str(target_path))
            print(f"Copied preprocessor_config.json from {config_file} to {target_path}")
        else:
            print(f"Warning: {config_file} does not exist and could not be copied.")
            
        if  "qwen2_vl" in str(base_model_path):
            preprocessor_config = load_json_file(target_path)
            preprocessor_config["max_pixels"] = 1605632
            dump_json_file(preprocessor_config,target_path,if_backup=False)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str,)
    parser.add_argument("--enable-processor", type=bool, default= True,)
    parser.add_argument("--final-model-path", type=str,)
    args = parser.parse_args()
    apply_full_model(args)
    