import os
import json
import pandas as pd

for dataset in ['type3']:
	os.makedirs(f'../../model_responses/{dataset}/', exist_ok=True)
	for image_type in ['simple', 'original', 'combined']:
		for prompt_type in ['zeroshot', 'zeroshotcot', 'directives']:
			if image_type == 'original' and (dataset == 'type3' or dataset == 'type2'):
				continue
			df = pd.read_json(f'../../{dataset}/qa.json')
			answer_df = df[['question', 'answer']] if dataset == 'type2' else df[['question_id', 'question', 'answer']]
			resps = []
			for i in range(len(os.listdir(f'GPT_batch_results/{dataset}/{image_type}/{prompt_type}'))):
				with open (f'GPT_batch_results/{dataset}/{image_type}/{prompt_type}/{i}.jsonl', 'r') as f:
					for line in f:
						resps.append(json.loads(line)['response']['body']['choices'][0]['message']['content'])
			
			answer_df['response'] = resps

			answer_df.to_json(f'../../model_responses/{dataset}/gpt4o_{image_type}_{prompt_type}.json', orient='records', indent=4)
			del df, answer_df, resps