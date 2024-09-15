import utils
import fasttext
import tempfile
import random
import copy
import numpy as np

random.seed(1)
np.random.seed(1)

source_filepath = '../data/monolingual/asturian'
target_filepath = '../data/monolingual/spanish'
source_dict_file = '../data/processed/dalla_dictionary.json'
target_dict_file = '../data/processed/rae_dictionary.json'
source_vector_file = '../fasttext_result/asturian.emb.txt'
target_vector_file = '../fasttext_result/spanish.emb.txt'
test_dict_file = '../data/test/test.json'

#TRAINING PARAMETERS
model_name = 'skipgram' #{skipgram, cbow}
loss_function = 'ns' #{ns, hs, softmax, ova}
vector_dim = 300
n_epoch = 7

"""Returns dictionary words"""
def get_dictionary_words(file):
    dictionary = utils.get_json_dictionary(file)
    words, _ = utils.get_dictionary_words_and_definitions(dictionary)
    return words

"""Delete definitions given dictionary keys or values"""
def delete_definitions_by_reference(words_list, definitions_list, reference):
    for i in range(len(words_list)-1, -1, -1):
        if words_list[i] in reference:
            del definitions_list[i]
            del words_list[i]
    return words_list, definitions_list

''' Saves fasttext word representations in Word2Vec format, which is needed in VecMap'''
def save_word_vectors(words, model, vector_file):
    n_words = len(list(set(words)))
    dim = model.get_dimension()
    with open(vector_file, mode='w', encoding='utf-8') as f:
        f.write(f"{n_words} {dim}\n")
        for word in list(set(words)):
            word2vec_format_word = word.replace(" ", "_")
            vector = model.get_word_vector(word)
            word2vec_format_vector = ' '.join(map(str, vector))
            f.write(f"{word2vec_format_word} {word2vec_format_vector}\n")
            
"""Trains fasttext model"""
def train_fasttext(file):
    model = fasttext.train_unsupervised(file, model=model_name, loss=loss_function, dim=vector_dim, epoch=n_epoch)
    return model

"""Creates temporal file with the data to use it in fasttext training"""
def create_data_file(data):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        for phrase in data:
            temp_file.write(phrase)
        temp_file_path = temp_file.name
    return temp_file_path

#CAMBIAR EN LATEX
def main():
    #Get training data
    source_filenames, target_filenames = utils.get_data_files(source_filepath), utils.get_data_files(target_filepath)
    source_data, target_data = utils.load_data(source_filenames), utils.load_data(target_filenames)
    #Get test set
    test_dict = utils.load_json_file(test_dict_file)
    source_words, source_definitions = utils.get_dictionary_words_and_definitions(utils.get_json_dictionary(source_dict_file))
    target_words, target_definitions = utils.get_dictionary_words_and_definitions(utils.get_json_dictionary(target_dict_file))
    #Delete samples if they are in test set
    source_words_copy, source_definitions_copy = copy.deepcopy(source_words), copy.deepcopy(source_definitions)
    target_words_copy, target_definitions_copy = copy.deepcopy(target_words), copy.deepcopy(target_definitions)
    delete_definitions_by_reference(source_words_copy, source_definitions_copy, test_dict.keys())
    delete_definitions_by_reference(target_words_copy, target_definitions_copy, test_dict.values())
    source_definitions_copy = list(set(source_definitions_copy))
    target_definitions_copy = list(set(target_definitions_copy))
    #Shuffle definitions
    random.shuffle(source_definitions_copy)
    random.shuffle(target_definitions_copy)
    #Set data training samples to the smaller one
    max_len = len(source_definitions_copy)
    target_definitions_copy = target_definitions_copy[:max_len]
    #Add definitions to the data
    source_data += source_definitions_copy
    target_data += target_definitions_copy
    #Create temp file
    source_training_file, target_training_file = create_data_file(source_data), create_data_file(target_data)
    #Train fasttext models
    source_model = train_fasttext(source_training_file)
    target_model = train_fasttext(target_training_file)
    # Save vocabulary vector representations
    save_word_vectors(source_words, source_model, source_vector_file)
    save_word_vectors(target_words, target_model, target_vector_file)
    
if __name__=='__main__':
    main()