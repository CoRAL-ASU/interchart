import pandas as pd
import os
from tqdm import tqdm
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_name = 'qwen2vl'
dataset = 'type1'
category = '1'
device = "cuda:0"

if dataset == 'type2':
    qas = pd.read_json(f'../../{dataset}/qa_{category}.json')
    image_indices = qas['chart_name'].values
else:
    qas = pd.read_json(f'../../{dataset}/qa.json')
    image_indices = qas['image_index'].values.astype(int)

if dataset == 'type1':
    image_indices = [f'multichart_{index}' for index in image_indices]
    
questions = qas['question'].values
answers = qas['answer'].values

tables_path = f'../../model_responses/chart_to_table/original/{model_name}'

def get_chart_info(index):
    all_info = json.load(open(f'{tables_path}/{index}_orig.json'))
    info_string = ''
    for i, info in enumerate(all_info):
        info_string += f'Information extracted from chart {i+1}:\n{info}\n\n'
        
    return info_string

prompt = """You are tasked with answering a specific question. The answer must be derived solely from information provided, which is extracted from images of charts. This information will include the data extracted from the chart, including the chart title. Your final answer to the question should strictly be in the format - \"Final Answer:\" <final_answer>. 

Let\'s work this out in a step by step way to be sure we have the right answer.

Data extracted from charts: 
{tables} 
Question: {question}"""

message_template = [
    {
        "role": "user",
        "content": []
    }
]

def get_message(image_index, prompt, question):
    message = message_template.copy()
    info = get_chart_info(image_index)
    message[0]['content'] = [{
        "type": "text",
        "text": prompt.format(tables=info, question=question)
    }]
    return message

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    cache_dir='/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache'
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

model_responses = []
     
def run_model():
    for i, question in enumerate(tqdm(questions)):
        image_index = image_indices[i] 
        messages = get_message(image_index, prompt, question)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
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
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': output_text[0].strip()})
        
if __name__ == '__main__':
    run_model()
    if dataset == 'type2':
        save_path = f'{dataset}_{category}'
    else:
        save_path = dataset
        
    os.makedirs(f'../../model_responses/table_qa/original/{save_path}', exist_ok=True)
    model_responses_df = pd.DataFrame(model_responses)
    model_responses_df.to_json(f'../../model_responses/table_qa/original/{save_path}/{model_name}.json', orient='records')