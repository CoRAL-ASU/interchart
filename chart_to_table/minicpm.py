from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:1'
model_name = "minicpm"

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',                                                    
    trust_remote_code=True,
    attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16,
    device_map = device,
    cache_dir='c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache') # sdpa or flash_attention_2, no eager

model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

types = ['t1', 't2', 't3']
directories = ['../../type1/simple', '../../type2/simple', '../../type3/simple']
input_prompt = """Your task is to extract all data from the chart image provided. Make sure to include the chart's title.
Output the data in a structured format. Ensure every data point is accurately captured and represented. Be meticulous and do not omit any information.

Think step by step. Identify the chart type to extract data accordingly."""

message_template = [
    {
        "role": "user",
        "content": []
    }
]

def get_message(image_path, prompt):
    images = [Image.open(image_path).convert('RGB')]
    message = message_template.copy()
    message[0]['content'] = images
    message[0]['content'].append({
        "type": "text",
        "text": prompt
    })
    return message

os.makedirs(f'../../model_responses/chart_to_table/{model_name}', exist_ok=True)

for i in range(len(types)):
    chart_type = types[i]
    directory = directories[i]
    
    all_charts = json.load(open(f'{chart_type}_charts.json'))    
    
    os.makedirs(f'../../model_responses/chart_to_table/{model_name}/{chart_type}', exist_ok=True)
    
    for key, value in tqdm(all_charts.items()):
        chart_name = key
        charts = value
        
        tables = []
        
        for chart in charts:
            image_path = f'{directory}/{chart}'
            messages = get_message(image_path, input_prompt)
        
            answer = model.chat(image=None, msgs=messages, tokenizer=tokenizer)
            extracted_table = answer.strip()
            tables.append(extracted_table)
            
        with open(f'../../model_responses/chart_to_table/{model_name}/{chart_type}/{chart_name}.json', 'w') as f:
            json.dump(tables, f, indent=4)