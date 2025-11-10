import pandas as pd
import os
import json
import base64
from PIL import Image
from tqdm import tqdm
import io
model_name = "gpt-4o-mini"
batch_size = 50
def get_prompt(prompt_type):

	if prompt_type == 'zeroshotcot':
		prompt = """Your task is the answer the question based on the given {img_word} Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\nLet's work this out in a step by step way to be sure we have the right answer.\n\nQuestion: {question}"""
	elif prompt_type == 'zeroshot':
		prompt = """Your task is the answer the question based on the given {img_word}. Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\n\nQuestion: {question}"""
	elif prompt_type == 'directives':
		prompt = """Your task is to answer a question based on a given {img_word}. To ensure clarity and accuracy, you are required to break down the question into steps of extraction and reasoning. Your final answer should strictly rely on the visual information presented in the {img_word}.

Here are a few directives that you can follow to reach your answer:

Step 1: Identify Relevant Entities
First, identify the key entities or data points needed to answer the given question. These could be labels, categories, values, or trends in the chart or image.

Example:
If you're given a chart showing GDP growth for five countries over several years, and the question asks for the difference between the highest GDP growth of Country A and Country B, the relevant entities are "Country A" and "Country B."

Step 2: Extract Relevant Values
Extract all necessary values related to the identified entities from the image. These values might be numerical (e.g., percentages, quantities) or categorical (e.g., labels, categories).

Example:
From the chart, extract all the values of GDP growth for both Country A and Country B.

Step 3: Reasoning and Calculation
Using the extracted values, apply logical reasoning and calculations to derive the correct answer. Clearly outline the reasoning process to ensure the steps leading to the final answer are understandable and correct.

Example:
Find the highest GDP growth for both Country A and Country B from the extracted values.
Calculate the difference between these highest values.

Step 4: Provide the Final Answer
Based on your reasoning, provide the final answer in the following format:
Final Answer: <final_answer>

Ensure that your reasoning is explicit and matches the information extracted from the {img_word}. Your answer should rely solely on the visual data provided, and you should reason step by step in order to ensure you reach the correct conclusion.

Question: {question}"""

	else:
		raise ValueError(f"Invalid prompt type: {prompt_type}")
	
	return prompt

def decode_image(base_64_string):
	image_data = base64.b64decode(base_64_string)
	image = Image.open(io.BytesIO(image_data))
	return image

def encode_image(image_path):
	with open(image_path, "rb") as image:
		return base64.b64encode(image.read()).decode("utf-8")

def format_payload(id, base64_images, prompt):
	
	content = [
		{
			"type": "image_url",
			"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}",
			}
		} for base64_image in base64_images
	]
	 
	content.append({
		"type": "text",
		"text": prompt
	})
	
	payload = {
		"custom_id": id,
		"method": "POST",
		"url": "/v1/chat/completions",
		"body": {
			"model": model_name,
			"messages": [
				{
					"role": "user",
					"content": content
				}
			],
			"max_tokens": 1000
		}
	}
	 
	return payload

def get_message(image_index, prompt, question, dataset, image_type, prompt_type, img_word, index_to_images, image_base_path, q_id):
	
	id = f'{dataset}_{image_type}_{prompt_type}_{q_id}'
	images_in_payload = []
	for image in index_to_images[image_index]:
		image_path = f'{image_base_path}{image}'
		image = Image.open(image_path).convert('RGB')
		base64_image = encode_image(image_path)
		images_in_payload.append(base64_image)

	payload = format_payload(id, images_in_payload, prompt.format(img_word=img_word, question=question))
	return payload

def get_payloads(qas, image_base_path, dataset, image_type, prompt_type, img_word, index_to_images, prompt, category = 0):
	if dataset == 'type2':
		image_indices = qas['chart_name'].values
	else:
		image_indices = qas['image_index'].values.astype(int)
	questions = qas['question'].values
	answers = qas['answer'].values

	all_images = os.listdir(image_base_path)

	if dataset == 'type1':
		prefix = 'multichart_'
		if image_type == 'original' or image_type == 'simple':
			sep = '_'
		else:
			sep = '.'
   
	if dataset == 'type2':
		prefix = ''
		if image_type == 'combined':
			sep = '.'
		elif image_type == 'simple':
			sep = '_'
		else:
			print("not allowed for type2")
			exit()

	if dataset == 'type3':
		if (image_type == 'combined'):
			prefix = 'multichart_'
			sep = '.'
		elif (image_type == 'simple'):
			prefix = ''
			sep = '_'
		else:
			print("not allowed for type3")
			exit()

	for index in set(image_indices):
		for image in all_images:
			if image.startswith(f'{prefix}{index}{sep}'):
				if index not in index_to_images:
					index_to_images[index] = []
				index_to_images[index].append(image)
	
		
	payloads = []
	for i in tqdm(range(len(image_indices))):
		payloads.append(get_message(image_index= image_indices[i], prompt=prompt, question=questions[i], dataset=dataset,
							  image_type=image_type, prompt_type=prompt_type, img_word=img_word, index_to_images=index_to_images, image_base_path=image_base_path, q_id = i))
		
	return questions, answers, payloads

for prompt_type in ['zeroshot', 'zeroshotcot', 'directives']:
	for dataset in ['type3']:
	# for dataset in ['type1', 'type2']:
		for image_type in ['combined', 'simple']:
		# for image_type in ['combined', 'simple', 'original']:
			
			if (dataset == 'type3' or dataset =='type2') and image_type == 'original':
				continue

			index_to_images = {}
			image_base_path = f'../../{dataset}/{image_type}/'
			if image_type == 'simple':
				img_word = 'images'
			else:
				img_word = 'image'

			qa_path = 'qa.json'

			df = pd.read_json(f'../../{dataset}/{qa_path}')
			questions, answers, payloads = get_payloads(df, image_base_path, dataset, image_type, prompt_type, img_word, index_to_images, get_prompt(prompt_type))
			os.makedirs(f'GPT_batches/{dataset}/{image_type}/{prompt_type}', exist_ok=True)
			payloads_batched = [payloads[i:i+batch_size] for i in range(0, len(payloads), batch_size)]

			for i, batch in enumerate(payloads_batched):
				for query in batch:
					with open(f'GPT_batches/{dataset}/{image_type}/{prompt_type}/{i}.jsonl', 'a') as f:
						json.dump(query, f)
						f.write('\n')
			
			print(f'Created {len(payloads_batched)} batches for {dataset} {image_type} {prompt_type} and cleared memory')
			del df, index_to_images, questions, answers, payloads, payloads_batched
