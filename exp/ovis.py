import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM

model_name = 'ovis'
prompt_type = 'zeroshot'
dataset = 'type1'
image_type = 'original'
device = "cuda:1"

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

if image_type == 'simple':
    img_word = 'images'
else:
    img_word = 'image'


def get_message(image_index, prompt, question):
    images = []
    for image in index_to_image[image_index]:
        image_path = f'{image_base_path}{image}'
        image = Image.open(image_path).convert('RGB')
        images.append(image)
    message = f'<image>\n{prompt.format(img_word=img_word, question=question)}'
    return message, images

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True, 
                                             cache_dir='/media/vivek/drive/multichartqa/models_cache',
                                             device_map=device).to(device)
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

model_responses = []
     
if prompt_type == 'zeroshotcot':
    prompt = """Your task is the answer the question based on the given {img_word} Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\nLet's work this out in a step by step way to be sure we have the right answer.\n\nQuestion: {question}"""
elif prompt_type == 'zeroshot':
    prompt = """Your task is the answer the question based on the given {img_word}. Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\n\nQuestion: {question}"""

def run_model():
    for i, question in enumerate(tqdm(questions)):
        image_index = image_indices[i] 
        message, images = get_message(image_index, prompt, question)
        text, input_ids, pixel_values = model.preprocess_inputs(message, images)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': output_text.strip()})
        print(f'Question: {question}\nGold: {answers[i]}\nResponse: {output_text.strip()}\n')
        
if __name__ == '__main__':
    run_model()
    os.makedirs(f'../model_responses/{dataset}', exist_ok=True)
    model_responses_df = pd.DataFrame(model_responses)
    model_responses_df.to_json(f'../model_responses/{dataset}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
