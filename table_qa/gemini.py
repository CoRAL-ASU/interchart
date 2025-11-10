import google.generativeai as genai
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import json

model_name = "gemini_1.5_pro"
table_giver = "deplot++"
dataset = 'type1'
category = '1'

genai.configure(api_key= '<api_key>') 
generation_config = {
  "temperature": 1,
  "max_output_tokens": 2048,
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]
model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings, generation_config=generation_config)

print(model.generate_content("Test"))

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

tables_path = f'../../model_responses/chart_to_table/original/{table_giver}'

if table_giver == 'deplot++':
    tables_path = f'../../model_responses/chart_to_table/original/deplot'
    titles_path = f'../../model_responses/captions/gemini/original/gemini'

def get_chart_info(index):
    all_info = json.load(open(f'{tables_path}/{index}_orig.json'))
    if table_giver == 'deplot++':
        chart_title = json.load(open(f'{titles_path}/{index}_orig.json'))
    info_string = ''
    for i, info in enumerate(all_info):
        if table_giver == 'deplot++':
            info_string += f'Information extracted from chart {i+1}:\n{chart_title[i]}:\n{info}\n\n'
        else:
            info_string += f'Information extracted from chart {i+1}:\n{info}\n\n'
        
    return info_string

prompt = """You are tasked with answering a specific question. The answer must be derived solely from information provided, which is extracted from images of charts. This information will include the data extracted from the chart, including the chart title. Your final answer to the question should strictly be in the format - \"Final Answer:\" <final_answer>. 

Let\'s work this out in a step by step way to be sure we have the right answer.

Data extracted from charts: 
{tables} 
Question: {question}"""

def get_message(image_index, prompt, question):
    info = get_chart_info(image_index)
    return prompt.format(tables=info, question=question)

def get_queries():    
    queries = []
    for i, question in enumerate(tqdm(questions)):
        query = get_message(image_indices[i], prompt, question)
        queries.append(query)        
    return queries  

def get_results(queries, max_workers=20):
    with ThreadPoolExecutor() as executor:
        executor._max_workers = max_workers
        results = list(tqdm(executor.map(generate_content, queries), total=len(queries)))
    return results

def generate_content(query):
    try:
        resp = model.generate_content(query)
        return resp.text
    except Exception as e:
        print(query, e)
        return 'Error by gemini'

queries = get_queries()
results = get_results(queries, max_workers=20)

results_output = []

for i in range(len(questions)):
    results_output.append({'question_id': i, 'question': questions[i], 'gold': answers[i], 'response': results[i]})

if dataset == 'type2':
    save_path = f'{dataset}_{category}'
else:
    save_path = dataset
os.makedirs(f'../../model_responses/table_qa/original/{save_path}', exist_ok=True)
model_responses_df = pd.DataFrame(results_output)
model_responses_df.to_json(f'../../model_responses/table_qa/original/{save_path}/{table_giver}.json', orient='records', indent=4)