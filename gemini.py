import google.generativeai as genai
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

# model_name = "gemini_1.5_pro"
# prompt_type = 'directives'
# dataset = 'type2'
# category = '1'
# image_type = 'simple'

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
#         # print(".", end="")
#         return resp.text
#     except Exception as e:
#         print(query, e)
#         return 'Error by gemini'

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

# index_to_image = {}
# image_base_path = f'../{dataset}/{image_type}/'
# if image_type == 'simple':
#     img_word = 'images'
# else:
#     img_word = 'image'
    
# def get_message(image_index, prompt, question):
#     query = []
#     for image in index_to_image[image_index]:
#         image_path = f'{image_base_path}{image}'
#         image = Image.open(image_path).convert('RGB')
#         query.append(image)
        
#     query.append(prompt.format(img_word=img_word, question=question))
#     return query

# def get_queries(qas):
#     if dataset == 'type2':
#         image_indices = qas['chart_name'].values
#     else:
#         image_indices = qas['image_index'].values.astype(int)
#     questions = qas['question'].values
#     answers = qas['answer'].values
    

#     all_images = os.listdir(image_base_path)

#     if dataset == 'type1':
#         prefix = 'multichart_'
#         if image_type == 'original' or image_type == 'simple':
#             sep = '_'
#         elif image_type == 'combined':
#             sep = '.'
#         else:
#             print("not allowed for type1")
#             exit()

#     if dataset == 'type2':
#         prefix = ''
#         if image_type == 'combined':
#             sep = '.'
#         elif image_type == 'simple':
#             sep = '_'
#         else:
#             print("not allowed for type2")
#             exit()

#     if dataset == 'type3':
#         if (image_type == 'combined'):
#             prefix = 'multichart_'
#             sep = '.'
#         elif (image_type == 'simple'):
#             prefix = ''
#             sep = '_'
#         else:
#             print("not allowed for type3")
#             exit()

#     for index in set(image_indices):
#         for image in all_images:
#             if image.startswith(f'{prefix}{index}{sep}'):
#                 if index not in index_to_image:
#                     index_to_image[index] = []
#                 index_to_image[index].append(image)
    
#     queries = []
#     for i in tqdm(range(len(image_indices))):
#         queries.append(get_message(image_indices[i], prompt, questions[i]))
        
#     return questions, answers, queries
                
# def get_message(image_index, prompt, question):
#     query = []
#     for image in index_to_image[image_index]:
#         image_path = f'{image_base_path}{image}'
#         image = Image.open(image_path).convert('RGB')
#         query.append(image)
        
#     query.append(prompt.format(img_word=img_word, question=question))
#     return query

# if dataset == 'type2':
#     df = pd.read_json(f'../{dataset}/qa_{category}.json')
# else:
#     df = pd.read_json(f'../{dataset}/qa.json')
    
# questions, answers, queries = get_queries(df)
# results = get_results(queries, max_workers=20)

# results_output = []

# for i in range(len(questions)):
#     results_output.append({
#         'question_id': i,
#         'question': questions[i],
#         'answer': answers[i],
#         'response': results[i]
#     })

# if dataset == 'type2':
#     save_path = f'{dataset}_{category}'
# else:
#     save_path = dataset
# os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
# model_responses_df = pd.DataFrame(results_output)
# model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records', indent=4)


model_name = "gemini_1.5_pro"
# prompt_type = 'directives'
dataset = 'type3'
category = '1'
# image_type = 'simple'

for prompt_type in ['zeroshotcot', 'zeroshot', 'directives']:
    for image_type in ['simple', 'combined']:

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
                # print(".", end="")
                return resp.text
            except Exception as e:
                print(query, e)
                return 'Error by gemini'

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

        index_to_image = {}
        image_base_path = f'../{dataset}/{image_type}/'
        if image_type == 'simple':
            img_word = 'images'
        else:
            img_word = 'image'
            
        def get_message(image_index, prompt, question):
            query = []
            for image in index_to_image[image_index]:
                image_path = f'{image_base_path}{image}'
                image = Image.open(image_path).convert('RGB')
                query.append(image)
                
            query.append(prompt.format(img_word=img_word, question=question))
            return query

        def get_queries(qas):
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
            
            queries = []
            for i in tqdm(range(len(image_indices))):
                queries.append(get_message(image_indices[i], prompt, questions[i]))
                
            return questions, answers, queries
                        
        def get_message(image_index, prompt, question):
            query = []
            for image in index_to_image[image_index]:
                image_path = f'{image_base_path}{image}'
                image = Image.open(image_path).convert('RGB')
                query.append(image)
                
            query.append(prompt.format(img_word=img_word, question=question))
            return query

        if dataset == 'type2':
            df = pd.read_json(f'../{dataset}/qa_{category}.json')
        else:
            df = pd.read_json(f'../{dataset}/qa.json')
            
        questions, answers, queries = get_queries(df)
        results = get_results(queries, max_workers=20)

        results_output = []

        for i in range(len(questions)):
            results_output.append({
                'question_id': i,
                'question': questions[i],
                'answer': answers[i],
                'response': results[i]
            })

        if dataset == 'type2':
            save_path = f'{dataset}_{category}'
        else:
            save_path = dataset
        os.makedirs(f'../model_responses/{save_path}', exist_ok=True)
        model_responses_df = pd.DataFrame(results_output)
        model_responses_df.to_json(f'../model_responses/{save_path}/{model_name}_{image_type}_{prompt_type}.json', orient='records', indent=4)