from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
import json
import torch
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

device = 'cuda:0'
model_name = "internvl"

path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    cache_dir = '/media/vivek/c33fd89b-a307-4208-a045-64d021572535/multichartqa/models_cache',
    device_map=device
    ).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=True)

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

types = ['t1', 't2', 't3']
directories = ['../../type1/simple', '../../type2/simple', '../../type3/simple']
input_prompt = """Your task is to extract all data from the chart image provided. Make sure to include the chart's title.
Output the data in a structured format. Ensure every data point is accurately captured and represented. Be meticulous and do not omit any information.

Think step by step. Identify the chart type to extract data accordingly."""

def get_message(image_path, prompt):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    message = f'<image>\n{prompt}'   
    return message, pixel_values

os.makedirs(f'../../model_responses/chart_to_table/{model_name}', exist_ok=True)

for i in range(len(types)):
    chart_type = types[i]
    directory = directories[i]
    
    all_charts = json.load(open(f'{chart_type}_charts.json'))    
    
    os.makedirs(f'../../model_responses/chart_to_table/{model_name}/{chart_type}', exist_ok=True)
    
    for key, value in tqdm(all_charts.items()):
        chart_name = key
        charts = value
        
        tables = []
        
        for chart in charts:
            image_path = f'{directory}/{chart}'
            message, pixel_values = get_message(image_path, input_prompt)
        
            response = model.chat(tokenizer, pixel_values, message, generation_config)            
            extracted_table = response.strip()
            tables.append(extracted_table)
            
        with open(f'../../model_responses/chart_to_table/{model_name}/{chart_type}/{chart_name}.json', 'w') as f:
            json.dump(tables, f, indent=4)