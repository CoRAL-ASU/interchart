from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = "deplot"

processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)

os.makedirs(f'../../../model_responses/chart_to_table/original/{model_name}', exist_ok=True)

chart_type = 't1'
directory = '../../../type1/original'

all_charts = json.load(open(f'complex_charts.json'))    

for chart in tqdm(all_charts):    
    image = Image.open(f'{directory}/{chart}.png')
    text = "Generate underlying data table of the figure below:"
    inputs = processor(image, text, return_tensors="pt", padding=True).to(device)    
    preds = model.generate(**inputs, max_new_tokens = 1024)
    tables = processor.batch_decode(preds, skip_special_tokens=True)

    with open(f'../../../model_responses/chart_to_table/original/{model_name}/{chart}.json', 'w') as f:
        json.dump(tables, f, indent=4)
