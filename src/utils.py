import os
import json

"""Get path filenames"""
def get_data_files(filepath):
    data_files = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith('.txt'):
                data_files.append(os.path.join(root, file))
    return data_files

"""Load filename data"""
def load_data(file_names):
    text = []
    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as file:
            text += file.readlines()
    return text

"""Get JSON dictionary"""
def get_json_dictionary(file):
    with open(file, 'r', encoding='utf-8') as f:
        dictionary = [tuple(item) for item in json.load(f)]
    return dictionary

"""Load JSON data"""
def load_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

"""Get dictionary words and definitions"""
def get_dictionary_words_and_definitions(dictionary):
    words, definitions = zip(*dictionary)
    words = list(words)
    definitions = list(definitions)
    return words, definitions