from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)

types = ['t1', 't2', 't3']
directories = ['../../type1/simple', '../../type2/simple', '../../type3/simple']

os.makedirs('../../model_responses/chart_to_table/deplot', exist_ok=True)

for i in range(len(types)):
    chart_type = types[i]
    directory = directories[i]
    
    all_charts = json.load(open(f'{chart_type}_charts.json'))    
    
    os.makedirs(f'../../model_responses/chart_to_table/deplot/{chart_type}', exist_ok=True)
    
    for key, value in tqdm(all_charts.items()):
        chart_name = key
        charts = value
        
        images = [Image.open(f'{directory}/{chart}') for chart in charts]
        texts = ["Generate underlying data table of the figure below:"] * len(images)
        
        inputs = processor(images, texts, return_tensors="pt", padding=True).to(device)
        preds = model.generate(**inputs, max_new_tokens = 1024)
        tables = processor.batch_decode(preds, skip_special_tokens=True)
    
        with open(f'../../model_responses/chart_to_table/deplot/{chart_type}/{chart_name}.json', 'w') as f:
            json.dump(tables, f, indent=4)

        del images, texts, inputs, preds, tables 