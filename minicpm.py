import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
import torch

# model_name = 'minicpm'
# prompt_type = 'zeroshotcot'
# dataset = 'type2'
# category = '1'
# image_type = 'simple'
# device = "cuda:1"

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
#         images.append(Image.open(image_path).convert('RGB'))
#     message = message_template.copy()
#     message[0]['content'] = images
#     message[0]['content'].append(prompt.format(img_word=img_word, question=question))
#     return message

# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',                                                    
#     trust_remote_code=True,
#     attn_implementation='flash_attention_2', 
#     torch_dtype=torch.bfloat16,
#     device_map = device,
#     cache_dir='/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache') # sdpa or flash_attention_2, no eager

# model = model.eval()
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

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

# Example:
# If you're given a chart showing GDP growth for five countries over several years, and the question asks for the difference between the highest GDP growth of Country A and Country B, the relevant entities are "Country A" and "Country B."

# Step 2: Extract Relevant Values
# Extract all necessary values related to the identified entities from the image. These values might be numerical (e.g., percentages, quantities) or categorical (e.g., labels, categories).

# Example:
# From the chart, extract all the values of GDP growth for both Country A and Country B.

# Step 3: Reasoning and Calculation
# Using the extracted values, apply logical reasoning and calculations to derive the correct answer. Clearly outline the reasoning process to ensure the steps leading to the final answer are understandable and correct.

# Example:
# Find the highest GDP growth for both Country A and Country B from the extracted values.
# Calculate the difference between these highest values.

# Step 4: Provide the Final Answer
# Based on your reasoning, provide the final answer in the following format:
# Final Answer: <final_answer>

# Ensure that your reasoning is explicit and matches the information extracted from the {img_word}. Your answer should rely solely on the visual data provided, and you should reason step by step in order to ensure you reach the correct conclusion.

# Question: {question}"""

# def run_model():
#     for i, question in enumerate(tqdm(questions)):
#         image_index = image_indices[i]
#         messages = get_message(image_index, prompt, question)
#         answer = model.chat(image=None, msgs=messages, tokenizer=tokenizer)
#         model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': answer.strip()})
#         # print(f'Question: {question}\nAnswer: {answers[i]}\nResponse: {answer.strip()}\n\n')
        
# if __name__ == '__main__':
#     run_model()
#     if dataset == 'type2':
#         save_path = f'{dataset}_{category}'
#     else:
#         save_path = dataset
#     os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
#     model_responses_df = pd.DataFrame(model_responses)
#     model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')

model_name = 'minicpm'
# prompt_type = 'zeroshotcot'
dataset = 'type3'
category = '1'
# image_type = 'simple'
device = "cuda:1"

for prompt_type in ['zeroshotcot', 'zeroshot', 'directives']:
    for image_type in ['combined', 'simple']:
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
                images.append(Image.open(image_path).convert('RGB'))
            message = message_template.copy()
            message[0]['content'] = images
            message[0]['content'].append(prompt.format(img_word=img_word, question=question))
            return message

        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',                                                    
            trust_remote_code=True,
            attn_implementation='flash_attention_2', 
            torch_dtype=torch.bfloat16,
            device_map = device,
            cache_dir='/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache') # sdpa or flash_attention_2, no eager

        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

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

        def run_model():
            for i, question in enumerate(tqdm(questions)):
                image_index = image_indices[i]
                messages = get_message(image_index, prompt, question)
                answer = model.chat(image=None, msgs=messages, tokenizer=tokenizer)
                model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': answer.strip()})
                # print(f'Question: {question}\nAnswer: {answers[i]}\nResponse: {answer.strip()}\n\n')
                
        
        run_model()
        if dataset == 'type2':
            save_path = f'{dataset}_{category}'
        else:
            save_path = dataset
        os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
        model_responses_df = pd.DataFrame(model_responses)
        model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
        
        del model, tokenizer, model_responses, model_responses_df, qas, image_indices, questions, answers, image_base_path, all_images, index_to_image, message_template, img_word, prompt, save_path