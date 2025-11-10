from openai import OpenAI
import os
import json
from time import time, strftime, localtime


client = OpenAI(api_key = "<api_key>")

batch_ids = {}

#  get current time
# current_time = time()
format_into_time = "%Y-%m-%d_%H-%M-%S"
current_time = time()
current_time = strftime(format_into_time, localtime(current_time))


for prompt_type in ['zeroshot']:
# for prompt_type in ['zeroshot', 'zeroshotcot', 'directives']:
	for dataset in ['type3']:
	# for dataset in ['type1', 'type2']:
		for image_type in ['simple', 'combined']:
			# skip till type2 zershotcot simple
			if (prompt_type == 'zeroshotcot' and dataset == 'type1'):
				continue
			if (prompt_type == 'zeroshotcot' and dataset == 'type2' and image_type == 'simple'):
				continue
			
			if (dataset == 'type1' and image_type == 'simple' and prompt_type == 'zeroshot'):
				continue
			if (dataset == 'type3' or dataset == 'type2') and image_type == 'original':
				continue
			batches = os.listdir(f'GPT_batches/{dataset}/{image_type}/{prompt_type}')
			for i in range(len(batches)):
				batch_input_file = client.files.create(
					file=open(f"GPT_batches/{dataset}/{image_type}/{prompt_type}/{i}.jsonl", "rb"),
					purpose="batch"
				)
				batch = client.batches.create(
					input_file_id = batch_input_file.id,
					endpoint="/v1/chat/completions",
					completion_window="24h",
					metadata={
					"description": f"{dataset}_{image_type}_{prompt_type}_batch_{i}",
					}
				)
				if prompt_type not in batch_ids:
					batch_ids[prompt_type] = {}
				if dataset not in batch_ids[prompt_type]:
					batch_ids[prompt_type][dataset] = {}
				if image_type not in batch_ids[prompt_type][dataset]:
					batch_ids[prompt_type][dataset][image_type] = []
				batch_ids[prompt_type][dataset][image_type].append(batch.id)

				print(f"Created batch {batch.id} for {dataset}_{image_type}_{prompt_type}_batch_{i} with file {[i]}.jsonl")

with open(f'batch_ids_{current_time}.json', 'w') as f:
	json.dump(batch_ids, f)
