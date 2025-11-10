from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:0'
model_name = "qwen2vl"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    cache_dir='/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache'
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

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
    images = [
        {
            "type": "image",
            "image": image_path
        }
    ]
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
            message = get_message(image_path, input_prompt)
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            
            generated_ids = model.generate(
                **inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            extracted_table = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            tables.append(extracted_table)
            
        with open(f'../../model_responses/chart_to_table/{model_name}/{chart_type}/{chart_name}.json', 'w') as f:
            json.dump(tables, f, indent=4)