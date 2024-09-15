from sentence_transformers import SentenceTransformer, util
import torch
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
import json
import argparse
import utils

torch.manual_seed(1)

api_key = 'your_api_key'
openai_model_name = "gpt-4"
sentence_transformers_model_name = "sentence-transformers/LaBSE"

parser = argparse.ArgumentParser(description='Script that generates word alignments for two languages via edit distance, embedding distance or llm')
parser.add_argument(
    'technique', 
    choices=['edit_distance', 'embedding_distance', 'llm'],
    help='The technique used to get the alignments. Possible values: edit_distance, embedding_distance, llm'
)
args = parser.parse_args()

"""Edit distance alignment implementation"""
def get_words_by_edit_distance(source_dictionary, target_dictionary):
    # Get words
    source_words = np.array(list(set([tuple[0] for tuple in source_dictionary])))
    target_words = np.array(list(set([tuple[0] for tuple in target_dictionary])))

    # Get edit distance of every word in source vocabulary
    final_words = {}
    for source_word in tqdm(source_words, desc="Creating alignment dictionary using edit distance"):
        distances = np.array([Levenshtein.distance(source_word, target_word) for target_word in target_words])
        final_words[source_word] = target_words[np.argmin(distances)]
    return final_words

"""Cosine similarity alignment implementation"""
def get_words_by_embeddings_distance(source_dictionary, target_dictionary):
    # Get words and definitions from dictionaries
    source_words, source_definitions = utils.get_dictionary_words_and_definitions(source_dictionary)
    target_words, target_definitions = utils.get_dictionary_words_and_definitions(target_dictionary)

    model = SentenceTransformer(sentence_transformers_model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 16
    source_batches = len(source_definitions)//batch_size
    target_batches = len(target_definitions)//batch_size

    if len(source_definitions) % batch_size != 0:
        source_batches += 1
    if len(target_definitions) % batch_size != 0:
        target_batches += 1

    # Iterate over batches to get the definition representations
    with tqdm(total=source_batches+target_batches, desc="Encoding the dictionaries using BERT model") as p:
        source_embeddings = []
        for i in range(source_batches):
            embeddings = model.encode(source_definitions[i*batch_size:i*batch_size+batch_size], convert_to_tensor=True, device=device, batch_size=batch_size)
            source_embeddings.append(embeddings)
            p.update()
        source_embeddings = torch.cat(source_embeddings)

        target_embeddings = []
        for i in range(target_batches):
            embeddings = model.encode(target_definitions[i*batch_size:i*batch_size+batch_size], convert_to_tensor=True, device=device, batch_size=batch_size)
            target_embeddings.append(embeddings)
            p.update()
        target_embeddings = torch.cat(target_embeddings)

    del model
    torch.cuda.empty_cache()

    target_embeddings = target_embeddings.to(device)
    # Iterate over batches to get cosine similarity
    final_words = {word: [] for word in source_words}
    with tqdm(total=source_batches, desc="Creating alignment dictionary using embedding distance") as p:
        for i in range(source_batches):
            actual_words = source_words[i*batch_size:i*batch_size+batch_size]
            source_embeddings_batch = source_embeddings[i*batch_size:i*batch_size+batch_size].to(device)
            similarities = util.cos_sim(source_embeddings_batch, target_embeddings)
            _, indexes = torch.max(similarities, dim=1)
            indexes = indexes.tolist()
            for j, word in enumerate(actual_words):
                final_words[word].append(target_words[indexes[j]])
            p.update()

    return final_words

"""LLM alignment implementation"""
def get_words_by_llm(dictionary):
    words, definitions = utils.get_dictionary_words_and_definitions(dictionary)

    client = OpenAI(api_key=api_key)
    system_message = [{"role": "system", "content": "Quisiera saber qué palabra en español se correspondería a la palabra en asturiano y su definición. Ten en cuenta que, dado que son idiomas cercanos, es posible que las palabras y definiciones utilizadas sean similares en ambos idiomas. Solo necesito que me devuelvas las palabras en español (en minúsculas y sin punto final)."}]
    few_shot_messages = [{"role": "user", "content": "Definición en asturiano: Anticuáu, pasáu de moda.\n Palabra en asturiano: ranciu"}, {"role": "assistant", "content": "rancio"},
                         {"role": "user", "content": "Definición en asturiano: Aición y efeutu de refrescar.\n Palabra en asturiano: refrescamientu"}, {"role": "assistant", "content": "refrescamiento"},
                         {"role": "user", "content": "Definición en asturiano: Emitir un ruíu agudu [soplando y axuntando los llabios].\n Palabra en asturiano: xiblar"}, {"role": "assistant", "content": "silbar"},
                         {"role": "user", "content": "Definición en asturiano: Cachiparru, arácnidu [parásitu de cuerpu achapláu que chupa'l sangre].\n Palabra en asturiano: llarasca"}, {"role": "assistant", "content": "garrapata"}]
    
    final_words = {word: [] for word in words}
    for word, definition in zip(words, definitions):
        content = "Definición en asturiano: " + definition + "\n Palabra en asturiano: " + word
        user_message = [{"role": "user", "content": content}]
        final_messages = system_message + few_shot_messages + user_message

        response = client.chat.completions.create(
            model=openai_model_name,
            messages=final_messages
        )

        final_words[word].append(response.choices[0].message.content)
    return final_words


def main():
    # Get dictionary items
    with open('../data/processed/dalla_dictionary.json', 'r', encoding='utf-8') as f:
        dalla_dictionary = [tuple(item) for item in json.load(f)]
    with open('../data/processed/rae_dictionary.json', 'r', encoding='utf-8') as f:
        rae_dictionary = [tuple(item) for item in json.load(f)]

    if args.technique == 'edit_distance':
        dictionary = get_words_by_edit_distance(dalla_dictionary, rae_dictionary)
    elif args.technique == 'embedding_distance':
        dictionary = get_words_by_embeddings_distance(dalla_dictionary, rae_dictionary)
    elif args.technique == 'llm':
        with open('../data/test/test.json', 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        test_source_words = list(test_dataset.keys())
        test_dictionary = [item for item in dalla_dictionary if item[0] in test_source_words]
        dictionary = get_words_by_llm(test_dictionary)

    # Save dictionary in JSON
    with open('../data/alignments/'+args.technique+'_alignment.json', 'w') as file:
        json.dump(dictionary, file)

if __name__=="__main__":
    main()