import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

model_name = 'pixtral'
prompt_type = 'zeroshot'
dataset = 'type1'
image_type = 'original'
device = "cuda:0"

qas = pd.read_json(f'../{dataset}/qa.json')

image_indices = qas['image_index'].values.astype(int)
questions = qas['question'].values
answers = qas['answer'].values

image_base_path = f'../{dataset}/{image_type}/'

all_images = os.listdir(image_base_path)
index_to_image = {}

if dataset == 'type1':
    prefix = 'multichart_'
    if image_type == 'original' or image_type == 'simple':
        sep = '_'
    else:
        sep = '.'

for index in set(image_indices):
    for image in all_images:
        if image.startswith(f'{prefix}{index}{sep}'):
            if index not in index_to_image:
                index_to_image[index] = []
            index_to_image[index].append(image)

def get_message(image_index, prompt, question):
    image_paths = []
    for image in index_to_image[image_index]:
        image_path = f'{image_base_path}{image}'
        image_pixels = Image.open(image_path)
        image_paths.append(image_pixels)
    
    img_suffix = '[IMG]' * len(index_to_image[image_index])
    message = f'<s>[INST]{prompt.format(question)}\n{img_suffix}[/INST]'
    return message, image_paths

model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir='/multichartqa/models_cache', torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(model_id)

model_responses = []

def run_model():
    for i, question in enumerate(tqdm(questions)):
        image_index = image_indices[i]
        prompt = """Your task is the answer the question based on the given image. Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\n\nQuestion: {}""" 
        message, image_paths = get_message(image_index, prompt, question)
        inputs = processor(images=image_paths, text=message, return_tensors="pt").to(device)
        generate_ids = model.generate(**inputs, max_new_tokens=1024)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': output[0].strip()})
        
if __name__ == '__main__':
    run_model()
    os.makedirs(f'../model_responses/{dataset}', exist_ok=True)
    model_responses_df = pd.DataFrame(model_responses)
    model_responses_df.to_json(f'../model_responses/{dataset}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
