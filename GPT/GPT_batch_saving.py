from openai import OpenAI
import json
import os

client = OpenAI(api_key = "<api_key>")
batch_ids = json.load(open(f"batch_ids_2025-02-13_23-22-43.json","r"))

counter = 0
total = 0
for dataset in ['type3']:
# for dataset in ['type1', 'type2']:
	for image_type in ['simple', 'combined']:
		if (dataset == 'type3' or dataset == 'type2') and image_type == 'original':
			continue
		for prompt_type in ['zeroshot', 'zeroshotcot', 'directives']:			
			os.makedirs(f"GPT_batch_results/{dataset}/{image_type}/{prompt_type}", exist_ok=True)
			already_done = set([elem.split(".")[0] for elem in os.listdir(f"GPT_batch_results/{dataset}/{image_type}/{prompt_type}")])
			batches = batch_ids[prompt_type][dataset][image_type]
			for batch_id in batches:
				total += 1
				processed_batch = client.batches.retrieve(batch_id)
				file_num = processed_batch.metadata['description'].split("_")[-1]
				if f"{file_num}" in already_done:
					counter += 1
					continue
				if processed_batch.status == "completed":
					counter += 1
					print(f"Downloading batch_id: {batch_id}")
					content = client.files.content(processed_batch.output_file_id)
					content.write_to_file(f"GPT_batch_results/{dataset}/{image_type}/{prompt_type}/{file_num}.jsonl")
				if processed_batch.status == "in_progress":
					# counter += 1
					print(f"In progress batch_id: {batch_id}")
					# content = client.files.content(processed_batch.output_file_id)
					# content.write_to_file(f"GPT_batch_results/{dataset}/{image_type}/{prompt_type}/{file_num}.jsonl")

print(f"Downloaded {counter} files out of {total}")