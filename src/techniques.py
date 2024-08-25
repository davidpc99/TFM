from sentence_transformers import SentenceTransformer, util
import editdistance
import pandas as pd
import torch
from openai import OpenAI
import numpy as np

import time
import os

rae_pd = pd.read_json('../data/json/rae.json', orient='records')
dalla_pd = pd.read_json('../data/json/dalla.json', orient='records')

rae_masculine = rae_pd['Masculine'].to_list()
rae_femenine = [word for word in rae_pd['Femenine'].to_list() if word is not None]
rae_definitions = rae_pd['Definitions'].to_list()

# Probar con bonitu, algo mal en los headwords
def get_words_by_edit_distance(input_words):
    words = rae_masculine + rae_femenine

    final_words = []
    for input_word in input_words:
        eval_values = [editdistance.eval(input_word, word) for word in words]
        final_words.append(words[eval_values.index(min(eval_values))])
    return final_words

#def get_embeddings_distance(entry):
#    model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
#    definitions_list = rae_pd['Definitions'].to_list()
#    masculine_words = rae_pd['Masculine'].to_list()
#
#    entry_embedding = model.encode(entry, convert_to_tensor=True)
#
#    current_best_similarity = 0
#    for idx, definitions in enumerate(definitions_list):
#        embeddings = model.encode(definitions, convert_to_tensor=True)
#        similarities = [util.cos_sim(entry_embedding, embedding) for embedding in embeddings]
#        if max(similarities) > current_best_similarity:
#            current_best_similarity = max(similarities)
#            #max_idx = similarities.index(max(similarities))
#            best_idx = idx
#        print(idx)
#    return masculine_words[best_idx]

def get_words_by_embeddings_distance(input_definitions, dictionary):
    model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
    
    #model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    tiempo = time.time()
    entry_embeddings = model.encode(input_definitions, convert_to_tensor=True, device=device)
    entry_embeddings = entry_embeddings.to(device)

    definitions = [item[1] for item in dictionary]
    embeddings = model.encode(definitions, convert_to_tensor=True)
    embeddings = embeddings.to(device)
    similarities_1 = util.cos_sim(entry_embeddings, embeddings)
    print(time.time()-tiempo)

    _, indexes = torch.max(similarities_1, dim=1)
    indexes = indexes.tolist()
    final_words = [dictionary[index] for index in indexes]
    return final_words
    final_words = []
    for similarity in similarities_1:
        index = torch.argmax(similarity).item()
        #index = similarity.tolist().index(max(similarity))
        final_words.append(dictionary[index][0])
        print(dictionary[index])
    print(time.time()-tiempo)
    return 

def get_words_by_llm(words, definitions, api_key):
    client = OpenAI(api_key=api_key)

    system_message = [{"role": "system", "content": "Quisiera saber qué palabra en español se correspondería a la palabra en asturiano y su definición. Ten en cuenta que, dado que son idiomas cercanos, es posible que las palabras y definiciones utilizadas sean similares en ambos idiomas. Solo necesito que me devuelvas las palabras en español (en minúsculas y sin punto final)."}]
    few_shot_messages = [{"role": "user", "content": "Definición en asturiano: Anticuáu, pasáu de moda.\n Palabra en asturiano: ranciu"}, {"role": "assistant", "content": "rancio"},
                         {"role": "user", "content": "Definición en asturiano: Aición y efeutu de refrescar.\n Palabra en asturiano: refrescamientu"}, {"role": "assistant", "content": "refrescamiento"},
                         {"role": "user", "content": "Definición en asturiano: Emitir un ruíu agudu [soplando y axuntando los llabios].\n Palabra en asturiano: xiblar"}, {"role": "assistant", "content": "silbar"},
                         {"role": "user", "content": "Definición en asturiano: Cachiparru, arácnidu [parásitu de cuerpu achapláu que chupa'l sangre].\n Palabra en asturiano: llarasca"}, {"role": "assistant", "content": "garrapata"}]
    
    final_words = []
    for word, definition in zip(words, definitions):
        content = "Definición en asturiano: " + definition + "\n Palabra en asturiano: " + word
        user_message = [{"role": "user", "content": content}]
        final_messages = system_message + few_shot_messages + user_message

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=final_messages
        )

        final_words.append(response.choices[0].message.content)
    return final_words


def flatten_words_and_definitions():
    dictionary = []
    for word, definitions in zip(rae_masculine, rae_definitions):
        dictionary += [(word, definition) for definition in definitions]
    return dictionary


        


#acepciones = {
#    "ablativo": {
#        "genero": "m",
#        "definiciones": [
#            "Gram. Caso de la declinación latina y de otras lenguas indoeuropeas, cuya función principal es expresar la procedencia local o temporal, y en latín también las relaciones de situación, tiempo, modo, instrumento, materia, etc., que en español suelen expresarse anteponiendo al nombre alguna preposición, entre las cuales son las más frecuentes bajo, con, de, desde, en, por y sin.",
#            "Gram. Clase de construcción absoluta propia del latín, caracterizada porque sus dos elementos constitutivos figuran en ablativo. Establece alguna circunstancia con respecto a la oración a la que suele preceder con autonomía fónica."
#        ]
#    },
#    "ablativo2": {
#        "genero": "adj",
#        "definiciones": [
#            "Perteneciente o relativo a la ablación."
#        ],
#        "femenino": "ablativa"
#    }
#}

