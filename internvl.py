import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# model_name = 'internvl'
# prompt_type = 'directives'
# dataset = 'type2'
# category = '1'
# image_type = 'combined'
# device_map = "cuda:0"

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# def build_transform(input_size):
#     MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#     transform = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD)
#     ])
#     return transform

# def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
#     best_ratio_diff = float('inf')
#     best_ratio = (1, 1)
#     area = width * height
#     for ratio in target_ratios:
#         target_aspect_ratio = ratio[0] / ratio[1]
#         ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#         if ratio_diff < best_ratio_diff:
#             best_ratio_diff = ratio_diff
#             best_ratio = ratio
#         elif ratio_diff == best_ratio_diff:
#             if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                 best_ratio = ratio
#     return best_ratio

# def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#     orig_width, orig_height = image.size
#     aspect_ratio = orig_width / orig_height

#     # calculate the existing image aspect ratio
#     target_ratios = set(
#         (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
#         i * j <= max_num and i * j >= min_num)
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#     # find the closest aspect ratio to the target
#     target_aspect_ratio = find_closest_aspect_ratio(
#         aspect_ratio, target_ratios, orig_width, orig_height, image_size)

#     # calculate the target width and height
#     target_width = image_size * target_aspect_ratio[0]
#     target_height = image_size * target_aspect_ratio[1]
#     blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#     # resize the image
#     resized_img = image.resize((target_width, target_height))
#     processed_images = []
#     for i in range(blocks):
#         box = (
#             (i % (target_width // image_size)) * image_size,
#             (i // (target_width // image_size)) * image_size,
#             ((i % (target_width // image_size)) + 1) * image_size,
#             ((i // (target_width // image_size)) + 1) * image_size
#         )
#         # split the image
#         split_img = resized_img.crop(box)
#         processed_images.append(split_img)
#     assert len(processed_images) == blocks
#     if use_thumbnail and len(processed_images) != 1:
#         thumbnail_img = image.resize((image_size, image_size))
#         processed_images.append(thumbnail_img)
#     return processed_images

# def load_image(image_file, input_size=448, max_num=12):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

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

# if image_type == 'simple':
#     img_word = 'images'
# else:
#     img_word = 'image'

# def get_message(image_index, prompt, question):
#     pixel_values = None
#     num_patches_list = []
#     for image in index_to_image[image_index]:
#         image_path = f'{image_base_path}{image}'
#         image_tensor = load_image(image_path, max_num=12).to(torch.bfloat16).to(device_map)
#         if pixel_values is None:
#             pixel_values = image_tensor
#         else:
#             pixel_values = torch.cat((pixel_values, image_tensor), dim=0)
#         num_patches_list.append(image_tensor.size(0))
    
#     img_prefix = '<image>\n' * len(index_to_image[image_index])
#     message = f'{img_prefix}{prompt.format(img_word=img_word, question=question)}'
#     return message, pixel_values, num_patches_list

# path = 'OpenGVLab/InternVL2-8B'
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True,
#     cache_dir = '/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache',
#     device_map=device_map
#     ).eval()
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
# generation_config = dict(max_new_tokens=1024, do_sample=True)

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
#         message, pixel_values, num_patches_list = get_message(image_index, prompt, question)
#         response = model.chat(tokenizer, pixel_values, message, generation_config, num_patches_list=num_patches_list)
        
#         model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': response.strip()})
#         # print(f'Question: {question}\nGold: {answers[i]}\nResponse: {response.strip()}\n\n')
        
# if __name__ == '__main__':
#     run_model()
#     if dataset == 'type2':
#         save_path = f'{dataset}_{category}'
#     else:
#         save_path = dataset
#     os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
#     model_responses_df = pd.DataFrame(model_responses)
#     model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
    
model_name = 'internvl'
# prompt_type = 'directives'
dataset = 'type3'
category = '1'
# image_type = 'combined'
device_map = "cuda:0"

for prompt_type in ['zeroshot', 'directives', 'zeroshotcot']:
    for image_type in ['combined', 'simple']:
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

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

        if image_type == 'simple':
            img_word = 'images'
        else:
            img_word = 'image'

        def get_message(image_index, prompt, question):
            pixel_values = None
            num_patches_list = []
            for image in index_to_image[image_index]:
                image_path = f'{image_base_path}{image}'
                image_tensor = load_image(image_path, max_num=12).to(torch.bfloat16).to(device_map)
                if pixel_values is None:
                    pixel_values = image_tensor
                else:
                    pixel_values = torch.cat((pixel_values, image_tensor), dim=0)
                num_patches_list.append(image_tensor.size(0))
            
            img_prefix = '<image>\n' * len(index_to_image[image_index])
            message = f'{img_prefix}{prompt.format(img_word=img_word, question=question)}'
            return message, pixel_values, num_patches_list

        path = 'OpenGVLab/InternVL2-8B'
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            cache_dir = '/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache',
            device_map=device_map
            ).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True)

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
                message, pixel_values, num_patches_list = get_message(image_index, prompt, question)
                response = model.chat(tokenizer, pixel_values, message, generation_config, num_patches_list=num_patches_list)
                
                model_responses.append({'question_id': i, 'question': question, 'gold': answers[i], 'response': response.strip()})
                # print(f'Question: {question}\nGold: {answers[i]}\nResponse: {response.strip()}\n\n')
                
        run_model()
        if dataset == 'type2':
            save_path = f'{dataset}_{category}'
        else:
            save_path = dataset
        os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
        model_responses_df = pd.DataFrame(model_responses)
        model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records')
        
        del model, tokenizer, model_responses, model_responses_df, qas, image_indices, questions, answers, index_to_image, all_images, path, prompt, save_path