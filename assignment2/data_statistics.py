
import json

file_name = './Data_Embedding/training.json'
with open(file_name) as file:
    json_file = json.load(file)

print(len(json_file))