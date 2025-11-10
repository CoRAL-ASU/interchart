import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# model_name = 'qwen2vl'
# prompt_type = 'zeroshot'
# dataset = 'type2'
# category = '1'
# image_type = 'simple'
# device = "cuda:0"

# if dataset == 'type2':
#     qas = pd.read_json(f'../{dataset}/qa_{category}.json')
#     image_indices = qas['chart_name'].values
# else:
#     qas = pd.read_json(f'../{dataset}/qa.json')
#     image_indices = qas['image_index'].values.astype(int)
# questions = qas['question'].values
# answers = qas['answer'].values

# image_base_path = f'../{dataset}/{image_type}/'

# all_images = os.listdir(image_base_path)
# index_to_image = {}

# if dataset == 'type1':
#     prefix = 'multichart_'
#     if image_type == 'original' or image_type == 'simple':
#         sep = '_'
#     elif image_type == 'combined':
#         sep = '.'
#     else:
#         print("not allowed for type1")
#         exit()

# if dataset == 'type2':
#     prefix = ''
#     if image_type == 'combined':
#         sep = '.'
#     elif image_type == 'simple':
#         sep = '_'
#     else:
#         print("not allowed for type2")
#         exit()

# if dataset == 'type3':
#     if (image_type == 'combined'):
#         prefix = 'multichart_'
#         sep = '.'
#     elif (image_type == 'simple'):
#         prefix = ''
#         sep = '_'
#     else:
#         print("not allowed for type3")
#         exit()

# for index in set(image_indices):
#     for image in all_images:
#         if image.startswith(f'{prefix}{index}{sep}'):
#             if index not in index_to_image:
#                 index_to_image[index] = []
#             index_to_image[index].append(image)

# message_template = [
#     {
#         "role": "user",
#         "content": []
#     }
# ]

# if image_type == 'simple':
#     img_word = 'images'
# else:
#     img_word = 'image'


# def get_message(image_index, prompt, question):
#     images = []
#     for image in index_to_image[image_index]:
#         image_path = f'{image_base_path}{image}'
#         images.append({
#             "type": "image",
#             "image": image_path
#         })
#     message = message_template.copy()
#     message[0]['content'] = images
#     message[0]['content'].append({
#         "type": "text",
#         "text": prompt.format(img_word=img_word, question=question)
#     })
#     return message

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map=device,
#     cache_dir='/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache'
# )
# model.eval()
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# model_responses = []
     
# if prompt_type == 'zeroshotcot':
#     prompt = """Your task is the answer the question based on the given {img_word} Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\nLet's work this out in a step by step way to be sure we have the right answer.\n\nQuestion: {question}"""
# elif prompt_type == 'zeroshot':
#     prompt = """Your task is the answer the question based on the given {img_word}. Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\n\nQuestion: {question}"""
# elif prompt_type == 'directives':
#     prompt = """Your task is to answer a question based on a given {img_word}. To ensure clarity and accuracy, you are required to break down the question into steps of extraction and reasoning. Your final answer should strictly rely on the visual information presented in the {img_word}.

# Here are a few directives that you can follow to reach your answer:

# Step 1: Identify Relevant Entities
# First, identify the key entities or data points needed to answer the given question. These could be labels, categories, values, or trends in the chart or image.

# Step 2: Extract Relevant Values
# Extract all necessary values related to the identified entities from the image. These values might be numerical (e.g., percentages, quantities) or categorical (e.g., labels, categories).

# Step 3: Reasoning and Calculation
# Using the extracted values, apply logical reasoning and calculations to derive the correct answer. Explicitly state the reasoning process to ensure the steps leading to the final answer are understandable and correct. Think step by step and make sure you arrive at the correct answer for the given question.

# Step 4: Provide the Final Answer
# Based on your reasoning, provide the final answer in the following format:
# Final Answer: <final_answer>

# Here's are a few examples of reasoning using the given directives:
# Example 1
# Chart Provided: You are shown a chart representing the monthly sales figures of four products (Product A, Product B, Product C, and Product D) across six months.
# Question: Which product had the highest average sales over the six months?

# Model's Response:
# Step 1: The relevant entities to focus on are the monthly sales figures for Product A, Product B, Product C, and Product D.
# Step 2: Extract the sales values for each product across all six months from the chart.
# Step 3: Calculate the average sales for each product by summing the sales values across the six months and dividing by six. Compare the averages to determine which product had the highest average sales.
# Step 4: Final Answer: The product with the highest average sales is <Product X>.

# Example 2
# Charts Provided: You are shown three separate charts, each displaying the temperature trends for City X, City Y, and City Z over a 12-month period.
# Question: Which city experienced the highest temperature drop between any two consecutive months, and what was the value of that drop?

# Model's Response:
# Step 1: The relevant entities are the monthly temperature values for each city (City X, City Y, and City Z) across their respective charts.
# Step 2: Extract the monthly temperature values for City X from its chart, for City Y from its chart, and for City Z from its chart.
# Step 3: For each city, calculate the temperature drop between every two consecutive months by subtracting the temperature of the later month from the earlier one. Compare the largest temperature drop for each city across the three charts. Identify the city with the greatest temperature drop and note the value.
# Step 4: Final Answer: The city that experienced the highest temperature drop is <City X/City Y/City Z>, and the value of that drop is <temperature_drop_value>.

# Ensure that your reasoning is explicit and matches the information extracted from the {img_word}. Your answer should rely solely on the visual data provided, and you should reason step by step in order to ensure you reach the correct conclusion.

# Question: {question}"""

# def run_model():
#     for i, question in enumerate(tqdm(questions)):
#         image_index = image_indices[i] 
#         messages = get_message(image_index, prompt, question)
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to(device)
        
#         generated_ids = model.generate(
#             **inputs, max_new_tokens=1024)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': output_text[0].strip()})
#         # print(f'Question: {question}\nAnswer: {answers[i]}\nResponse: {output_text[0].strip()}\n\n')
        
# if __name__ == '__main__':
#     run_model()
#     if dataset == 'type2':
#         save_path = f'{dataset}_{category}'
#     else:
#         save_path = dataset
#     os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
#     model_responses_df = pd.DataFrame(model_responses)
#     model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
    
model_name = 'qwen2vl'
# prompt_type = 'zeroshot'
dataset = 'type3'
category = '1'
# image_type = 'simple'
device = "cuda:0"

for prompt_type in ['zeroshot', 'zeroshotcot', 'directives']:
    for image_type in ['simple', 'combined']:
        if dataset == 'type2':
            qas = pd.read_json(f'../{dataset}/qa_{category}.json')
            image_indices = qas['chart_name'].values
        else:
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
            elif image_type == 'combined':
                sep = '.'
            else:
                print("not allowed for type1")
                exit()

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
                    if index not in index_to_image:
                        index_to_image[index] = []
                    index_to_image[index].append(image)

        message_template = [
            {
                "role": "user",
                "content": []
            }
        ]

        if image_type == 'simple':
            img_word = 'images'
        else:
            img_word = 'image'


        def get_message(image_index, prompt, question):
            images = []
            for image in index_to_image[image_index]:
                image_path = f'{image_base_path}{image}'
                images.append({
                    "type": "image",
                    "image": image_path
                })
            message = message_template.copy()
            message[0]['content'] = images
            message[0]['content'].append({
                "type": "text",
                "text": prompt.format(img_word=img_word, question=question)
            })
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
            
        if prompt_type == 'zeroshotcot':
            prompt = """Your task is the answer the question based on the given {img_word} Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\nLet's work this out in a step by step way to be sure we have the right answer.\n\nQuestion: {question}"""
        elif prompt_type == 'zeroshot':
            prompt = """Your task is the answer the question based on the given {img_word}. Your final answer to the question should strictly be in the format - "Final Answer:" <final_answer>.\n\nQuestion: {question}"""
        elif prompt_type == 'directives':
            prompt = """Your task is to answer a question based on a given {img_word}. To ensure clarity and accuracy, you are required to break down the question into steps of extraction and reasoning. Your final answer should strictly rely on the visual information presented in the {img_word}.

        Here are a few directives that you can follow to reach your answer:

        Step 1: Identify Relevant Entities
        First, identify the key entities or data points needed to answer the given question. These could be labels, categories, values, or trends in the chart or image.

        Step 2: Extract Relevant Values
        Extract all necessary values related to the identified entities from the image. These values might be numerical (e.g., percentages, quantities) or categorical (e.g., labels, categories).

        Step 3: Reasoning and Calculation
        Using the extracted values, apply logical reasoning and calculations to derive the correct answer. Explicitly state the reasoning process to ensure the steps leading to the final answer are understandable and correct. Think step by step and make sure you arrive at the correct answer for the given question.

        Step 4: Provide the Final Answer
        Based on your reasoning, provide the final answer in the following format:
        Final Answer: <final_answer>

        Here's are a few examples of reasoning using the given directives:
        Example 1
        Chart Provided: You are shown a chart representing the monthly sales figures of four products (Product A, Product B, Product C, and Product D) across six months.
        Question: Which product had the highest average sales over the six months?

        Model's Response:
        Step 1: The relevant entities to focus on are the monthly sales figures for Product A, Product B, Product C, and Product D.
        Step 2: Extract the sales values for each product across all six months from the chart.
        Step 3: Calculate the average sales for each product by summing the sales values across the six months and dividing by six. Compare the averages to determine which product had the highest average sales.
        Step 4: Final Answer: The product with the highest average sales is <Product X>.

        Example 2
        Charts Provided: You are shown three separate charts, each displaying the temperature trends for City X, City Y, and City Z over a 12-month period.
        Question: Which city experienced the highest temperature drop between any two consecutive months, and what was the value of that drop?

        Model's Response:
        Step 1: The relevant entities are the monthly temperature values for each city (City X, City Y, and City Z) across their respective charts.
        Step 2: Extract the monthly temperature values for City X from its chart, for City Y from its chart, and for City Z from its chart.
        Step 3: For each city, calculate the temperature drop between every two consecutive months by subtracting the temperature of the later month from the earlier one. Compare the largest temperature drop for each city across the three charts. Identify the city with the greatest temperature drop and note the value.
        Step 4: Final Answer: The city that experienced the highest temperature drop is <City X/City Y/City Z>, and the value of that drop is <temperature_drop_value>.

        Ensure that your reasoning is explicit and matches the information extracted from the {img_word}. Your answer should rely solely on the visual data provided, and you should reason step by step in order to ensure you reach the correct conclusion.

        Question: {question}"""

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
                # print(f'Question: {question}\nAnswer: {answers[i]}\nResponse: {output_text[0].strip()}\n\n')
                
        run_model()
        if dataset == 'type2':
            save_path = f'{dataset}_{category}'
        else:
            save_path = dataset
        os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
        model_responses_df = pd.DataFrame(model_responses)
        model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
        
        del model, processor, model_responses, model_responses_df, qas, questions, answers, image_indices, image_base_path, all_images, index_to_image, message_template, img_word, prompt, save_path
            