from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:0'
model_name = "idefics"

processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16, cache_dir = 'c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache', device_map = device
)
model = model.eval()

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
    image = [Image.open(image_path).convert('RGB')]
    message = message_template.copy()
    message[0]['content'] = [
        {
            "type": "image"        
        },
        {
            "type": "text",
            "text": prompt
        }
    ]
    return message, image

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
            message, image = get_message(image_path, input_prompt)
        
            text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted_table = answer.strip()
            tables.append(extracted_table)
            
        with open(f'../../model_responses/chart_to_table/{model_name}/{chart_type}/{chart_name}.json', 'w') as f:
            json.dump(tables, f, indent=4)