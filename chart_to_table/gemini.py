import google.generativeai as genai
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import json

model_name = "gemini"

types = ['t1', 't2', 't3']
directories = ['../../type1/simple', '../../type2/simple', '../../type3/simple']

# # input_prompt = """Your task is to extract all data from the chart image provided. Make sure to include the chart's title.
# # Output the data in a structured format. Ensure every data point is accurately captured and represented. Be meticulous and do not omit any information.

# # Think step by step. Identify the chart type to extract data accordingly."""

# input_prompt = """Your task is to extract the main title of the chart image. The main title is typically located at the top of the chart, above the chart area itself, and describes the overall subject of the chart. The title usually describes what data is being presented, the time period, or the geographic location, if applicable. If the chart does not have a discernible main title, your response should be 'Title: None'. Otherwise, your response should be in the format 'Title: <title>'."""

# genai.configure(api_key= '<api_key>') 
# generation_config = {
#   "temperature": 1,
#   "max_output_tokens": 2048,
# }
# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_NONE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_NONE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_NONE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_NONE"
#   }
# ]
# model = genai.GenerativeModel('gemini-1.5-pro-002', safety_settings=safety_settings, generation_config=generation_config)

# print(model.generate_content("Test"))

# def get_results(queries, max_workers=20):
#     with ThreadPoolExecutor() as executor:
#         executor._max_workers = max_workers
#         results = list(tqdm(executor.map(generate_content, queries), total=len(queries)))
#     return results

# def generate_content(query):
#     try:
#         resp = model.generate_content(query)
#         return resp.text
#     except Exception as e:
#         print(query, e)
#         return 'Error by gemini'

# def get_message(image_path, prompt):
#     query = []
#     image = Image.open(image_path).convert('RGB')
#     query.append(image)
#     query.append(prompt)
#     return query

# os.makedirs(f'../../model_responses/captions/{model_name}', exist_ok=True)

# for i in range(len(types)):
#     chart_type = types[i]
#     directory = directories[i]
		
#     all_charts = json.load(open(f'{chart_type}_charts.json'))    
		
#     os.makedirs(f'../../model_responses/captions/{model_name}/{chart_type}', exist_ok=True)
		
#     all_chart_images = []
#     for key, value in tqdm(all_charts.items(), desc=f'Processing {chart_type}'):
#         all_chart_images.extend(value)
				
#     queries = []
#     for chart in tqdm(all_chart_images, desc=f'Generating queries for {chart_type}'):
#         image_path = f'{directory}/{chart}'
#         query = get_message(image_path, input_prompt)
#         queries.append(query)
				
#     print(queries[0])
		
#     results = get_results(queries, max_workers=5)
		
#     tables = {}
		
#     for i in range(len(all_chart_images)):
#         chart_name = all_chart_images[i]
#         chart_family = chart_name.rsplit('_', 1)[0]
#         if chart_family not in tables:
#             tables[chart_family] = []
#         tables[chart_family].append(results[i])
				
#     for key, value in tables.items():
#         with open(f'../../model_responses/captions/{model_name}/{chart_type}/{key}.json', 'w') as f:
#             json.dump(value, f, indent=4)
					
input_prompt = """Your task is to extract all data from the chart image provided. Make sure to include the chart's title.
Output the data in a structured format. Ensure every data point is accurately captured and represented. Be meticulous and do not omit any information.

Think step by step. Identify the chart type to extract data accordingly."""

# input_prompt = """Your task is to extract the main title of the chart image. The main title is typically located at the top of the chart, above the chart area itself, and describes the overall subject of the chart. The title usually describes what data is being presented, the time period, or the geographic location, if applicable. If the chart does not have a discernible main title, your response should be 'Title: None'. Otherwise, your response should be in the format 'Title: <title>'."""

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
model = genai.GenerativeModel('gemini-1.5-pro-002', safety_settings=safety_settings, generation_config=generation_config)

print(model.generate_content("Test"))

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

def get_message(image_path, prompt):
		query = []
		image = Image.open(image_path).convert('RGB')
		query.append(image)
		query.append(prompt)
		return query

os.makedirs(f'../../model_responses/chart_to_table/{model_name}', exist_ok=True)

chart_type = 't2'
directory = '../../type2/simple'

all_charts = json.load(open(f't2_left_charts.json'))    

os.makedirs(f'../../model_responses/chart_to_table/{model_name}/{chart_type}', exist_ok=True)

all_chart_images = []
for key, value in tqdm(all_charts.items(), desc=f'Processing {chart_type}'):
		all_chart_images.extend(value)
		
queries = []
for chart in tqdm(all_chart_images, desc=f'Generating queries for {chart_type}'):
		image_path = f'{directory}/{chart}'
		query = get_message(image_path, input_prompt)
		queries.append(query)
		
print(queries[0])

results = get_results(queries, max_workers=5)

tables = {}

for i in range(len(all_chart_images)):
		chart_name = all_chart_images[i]
		chart_family = chart_name.rsplit('_', 1)[0]
		if chart_family not in tables:
				tables[chart_family] = []
		tables[chart_family].append(results[i])
		
for key, value in tables.items():
		with open(f'../../model_responses/chart_to_table/{model_name}/{chart_type}/{key}.json', 'w') as f:
				json.dump(value, f, indent=4)
		print(f'Completed {key}')
