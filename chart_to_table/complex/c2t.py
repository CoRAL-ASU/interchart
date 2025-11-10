from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import json
import torch
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = "c2t"

model = VisionEncoderDecoderModel.from_pretrained("khhuang/chart-to-table").cuda()
processor = DonutProcessor.from_pretrained("khhuang/chart-to-table")

input_prompt = "<data_table_generation> <s_answer>"

os.makedirs(f'../../../model_responses/chart_to_table/original/{model_name}', exist_ok=True)

chart_type = 't1'
directory = '../../../type1/original'

all_charts = json.load(open(f'complex_charts.json'))    

for chart in tqdm(all_charts):
    tables = []
    img = Image.open(f'{directory}/{chart}.png')
    pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
    pixel_values = pixel_values.cuda()
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids.cuda()#.squeeze(0)

    outputs = model.generate(
        pixel_values.cuda(),
        decoder_input_ids=decoder_input_ids.cuda(),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    extracted_table = sequence.split("<s_answer>")[1].strip()
    tables.append(extracted_table)
    
    with open(f'../../../model_responses/chart_to_table/original/{model_name}/{chart}.json', 'w') as f:
        json.dump(tables, f, indent=4)