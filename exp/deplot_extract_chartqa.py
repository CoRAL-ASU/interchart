import pandas as pd
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import os
import json


processor = Pix2StructProcessor.from_pretrained('google/deplot', device_map="cuda")
deplot = Pix2StructForConditionalGeneration.from_pretrained('google/deplot', device_map="cuda")

chart_type = "complex"
question_type = "complex"
question_set = "a"

if question_set == "h":
    question_set = "test_human"
else:
    question_set = "test_augmented"

df = pd.read_json('../dataset/{}/{}_{}.json'.format(chart_type, question_set, question_type))
images = df['imgname'].tolist()
img_path = '../dataset/{}'.format(chart_type)
images = [img_path + '/png/' + img for img in images]
images = [Image.open(img) for img in images]

tables = []

for image in images:
    # deplot part of the pipeline 
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    inputs = inputs.to('cuda')
    predictions = deplot.generate(**inputs, max_new_tokens=2048)
    table = processor.decode(predictions[0], skip_special_tokens=True)
    print(table)

    tables.append(table)

os.makedirs('../tables/{}'.format(chart_type), exist_ok=True)

with open('../tables/{}/{}_{}.json'.format(chart_type, question_set, question_type), 'w') as f:
    json.dump(tables, f)