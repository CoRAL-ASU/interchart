import json
import pandas as pd

qa = pd.read_json('../../../type1/qa.json')
all_charts = qa['image_index'].unique()

all_charts = [f'multichart_{i}_orig' for i in all_charts]
json.dump(all_charts, open('complex_charts.json', 'w'))