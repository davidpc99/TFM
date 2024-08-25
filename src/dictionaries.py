from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re
import pandas as pd
import functools

DALLA_to_value = {'Sustantivu':'Sustantivo', 'Preposición':'Preposición', 'Interxección':'Interjección', 'Verbu':'Verbo', 'Axetivu':'Adjetivo',
                    'Alverbiu':'Adverbio', 'Conxunción':'Conjunción', 'Pronome':'Pronombre', 'Locución':'Locución'}

DRAE_to_value = {'interj.':'Interjección', 'loc.':'Locución', 'Loc.':'Locución', 'part.':'Verbo'}

def get_html_dictionary(xhtml_page):
    with open(xhtml_page, "r", encoding="utf-8") as dictionary_file:
        html_content = dictionary_file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    entries = soup.find_all('p', class_='asangre')
    entries_with_b = [entry for entry in entries if entry.find('b')]
    headwords = [entry.find('b').text for entry in entries_with_b]
    filtered_entries = [entry.text[len(headwords[idx])+3:] for idx, entry in enumerate(entries_with_b)]
    filtered_headwords = list(map(lambda x: x[:-1], headwords))

    return filtered_headwords, filtered_entries

def get_excel_dictionary(file_path):
    return pd.read_excel(file_path)

def get_meaning_numbers_and_positions(entry):
    regex = r"([1-9]\.|[1-9][0-9]\.)"
    meaning_numbers = re.findall(regex, entry)
    meaning_positions_information = re.finditer(regex, entry)
    meaning_positions = [(match.start(), match.end()) for match in meaning_positions_information]

    return meaning_numbers, meaning_positions

def get_dalla_meaning_numbers_and_positions(entry):
    regex = r"([1-9]|[1-9][0-9])"
    meaning_positions_information = re.finditer(regex, entry)
    meaning_positions = [(match.start(), match.end()) for match in meaning_positions_information]

    return meaning_positions if meaning_positions != [] else None

""" Returns the gendre suffixes of the headwords with the symbols ' -' """
def get_dalla_suffixes(entry):
    regex = r"\s-\w+"
    gendre_suffix = re.findall(regex, entry)
    return gendre_suffix

"""Returns the final gendre DALLA headwords given a headword and its suffixes"""
def get_dalla_gendre_headwords_given_suffixes(headword, suffixes):
    def is_vowel(character):
        vowel = set('aeiouáéíóú')
        return character in vowel

    if not suffixes:
        return (headword, None)
    else:
        for i in range(len(headword)-1, -1, -1):
            # Final gendre words are the headword until the last vocal is encountered plus the suffix without ' -'
            if is_vowel(headword[i]):
                if len(suffixes) == 1:
                    femenine_headword = headword[:i]+suffixes[0][2:]
                    masculine_headword = headword 
                else:
                    femenine_headword = headword[:i]+suffixes[0][2:]
                    masculine_headword = headword[:i]+suffixes[1][2:]
                return (masculine_headword, femenine_headword)
            
def get_rae_gendre_headwords_given_suffix(headword, suffix):
    def is_vowel(character):
        vowel = set('aeiouáéíóú')
        return character in vowel
    
    def is_consonant(character):
        consonant = set('bcdfghjklmnñpqrstvwxyz')
        return character in consonant
    
    if is_vowel(suffix[0]):
        find_function = is_vowel
    elif is_consonant(suffix[0]):
        find_function = is_consonant
    for i in range(len(headword)-1, -1, -1):
        if find_function(headword[i]):
            femenine_headword = headword[:i]+suffix
            masculine_headword = headword
            break
    return (masculine_headword, femenine_headword)




def filter_meaning_numbers_and_positions(meaning_numbers, meaning_positions):
    if len(meaning_numbers) == 0:
        return None
    filtered_meaning_numbers = [meaning_numbers[0]]
    filtered_meaning_positions = [meaning_positions[0]]
    for idx, i in enumerate(meaning_numbers[1:]):
        if int(i[0:-1]) > int(filtered_meaning_numbers[-1][0:-1]):
            filtered_meaning_numbers.append(i)
            filtered_meaning_positions.append(meaning_positions[idx+1])
    return filtered_meaning_positions

def get_types(entry, type_set):
    type_set.add(entry.split()[0])

##########################
####### Exceptions #######
##########################


def delete_exceptions(headwords, entries, numbers_and_positions, exception_indexes):
    exception_indexes = [index for index in range(len(numbers_and_positions)) if numbers_and_positions[index] == None]
    
    for index in sorted(exception_indexes, reverse=True):
        headwords.pop(index)
        entries.pop(index)
        numbers_and_positions.pop(index)
    
def delete_meaning_numbers(entry, meaning_positions):
        entry_without_numbers = []
        start = 0
        for start_end in meaning_positions:
            if start !=0:
                end = start_end[0]
                entry_without_numbers.append(entry[start+1:end-1])
            start =  start_end[1]
        entry_without_numbers.append(entry[start+1:])
        
        return entry_without_numbers

def get_locs(exception_entries):
    return [index for index, string in enumerate(exception_entries) if 'Loc.' in string or 'loc.' in string]

def get_parts(exception_entries):
    return [index for index, string in enumerate(exception_entries) if 'Part.' in string or 'part.' in string]

    
def main():
    DRAE_dictionaries_folder = "../data/dictionaries/DRAE/"
    DALLA_dictionary_folder = "../data/dictionaries/DALLA/"
    xhtml_pages = os.listdir(DRAE_dictionaries_folder)
    dictionay_xhtml_pages = list(filter(lambda x: x[:3]=="RAE" , xhtml_pages))

    print("Descargando la información de los diccionarios:")
    final_document_headwords = []
    final_document_entries = []
    document_exception_headwords = []
    document_exception_entries = []
    for dictionary in tqdm(dictionay_xhtml_pages):
        document_headwords, document_entries = get_html_dictionary(DRAE_dictionaries_folder+dictionary)
        document_meaning_number_positions = [filter_meaning_numbers_and_positions(*get_meaning_numbers_and_positions(entry)) for entry in document_entries]
        # Eliminar de las listas los que deben ser tratados manualmente (None, None)
        exception_indexes = [index for index in range(len(document_meaning_number_positions)) if document_meaning_number_positions[index] is None]
        # Eliminar las excepciones y guardarlas en otras listas para tratarlas de manera distinta
        document_exception_headwords.extend(document_headwords[index] for index in exception_indexes)
        document_exception_entries.extend(document_entries[index] for index in exception_indexes)
        #Eliminar excepciones de la lista de palabras
        delete_exceptions(document_headwords, document_entries, document_meaning_number_positions, exception_indexes)

        # Iterate through all word entries
        for idx, number_positions in enumerate(document_meaning_number_positions):
            # Saves all word entries without the numbers
            word_entries = []
            # Iterate through all entry positions
            for i in range(len(number_positions)):
                # Save new entry
                entry_start = number_positions[i][1]+1
                if i < len(number_positions)-1:
                    entry_end = number_positions[i+1][0]-1
                else:
                    entry_end = -1
                new_entry = document_entries[idx][entry_start:entry_end]
                if new_entry != '':
                    word_entries.append(new_entry)
            # Add headwords and new entries to the final list
            final_document_entries.append(word_entries)
        final_document_headwords.extend(document_headwords)
    
    document_exception_entries = [[entry] for entry in document_exception_entries]
    final_document_headwords = final_document_headwords + document_exception_headwords
    final_document_entries = final_document_entries + document_exception_entries

    for idx, headword in enumerate(final_document_headwords):
        splitted = headword.split(', ')
        if len(splitted) == 1:
            final_document_headwords[idx] = (headword, None)
        else:
            # Search for different headwords to keep first one. Head word example: 'áfilo, la o afilo, la'
            real_suffix = splitted[1].split()[0]
            final_document_headwords[idx] = get_rae_gendre_headwords_given_suffix(splitted[0], real_suffix)

    dalla_pd = get_excel_dictionary(DALLA_dictionary_folder+"DALLA.xls")
    headword_list = dalla_pd['Pallabra'].to_list()
    description_list = dalla_pd['Descripción'].to_list()
    entry_list = dalla_pd['Entrada'].to_list()      


    dalla_meaning_number_positions = [get_dalla_meaning_numbers_and_positions(entry) for entry in description_list]

    # Iterate through all word entries
    for idx, number_positions in enumerate(dalla_meaning_number_positions):
        # Saves all word entries without the numbers
        word_entries = []
        # If different entries
        if number_positions == None:
            word_entries.append(description_list[idx])
        else:
            # Iterate through all entry positions
            for i in range(len(number_positions)):
                # Save new entry
                entry_start = number_positions[i][1]+1
                if i == 0:
                    word_entries.append(description_list[idx][:entry_start-2])
                if i < len(number_positions)-1:
                    entry_end = number_positions[i+1][0]-1
                else:
                    entry_end = None
                new_entry = description_list[idx][entry_start:entry_end]
                if new_entry != '':
                    word_entries.append(new_entry)
        # Add headwords and new entries to the final list
        suffixes = get_dalla_suffixes(entry_list[idx])
        gendre_headwords = get_dalla_gendre_headwords_given_suffixes(headword_list[idx], suffixes)
        headword_list[idx] = gendre_headwords
        description_list[idx] = word_entries

    # Dataframe creation
    #df_dalla_headwords = pd.DataFrame({
    #    'ID': [i for i in range(len(headword_list))],
    #    'Masculine': [tuple[0] for tuple in headword_list],
    #    'Femenine': [tuple[1] for tuple in headword_list]
    #})
#
    #df_dalla_entries = pd.DataFrame({
    #    'ID': [i for i in range(len([description for word_descriptions in description_list for description in word_descriptions]))],
    #    'Description': [description for word_descriptions in description_list for description in word_descriptions], 
    #    'Headword_id': [idx for idx, word_descriptions in enumerate(description_list) for _ in word_descriptions]
    #})
#
    #df_rae_headwords = pd.DataFrame({
    #    'ID': [i for i in range(len(final_document_headwords))],
    #    'Masculine': [tuple[0] for tuple in final_document_headwords],
    #    'Femenine': [tuple[1] for tuple in final_document_headwords]
    #})
#
    #df_rae_entries = pd.DataFrame({
    #    'ID': [i for i in range(len([description for word_descriptions in final_document_entries for description in word_descriptions]))],
    #    'Description': [description for word_descriptions in final_document_entries for description in word_descriptions], 
    #    'Headword_id': [idx for idx, word_descriptions in enumerate(final_document_entries) for _ in word_descriptions]
    #})
#
    #df_dalla_headwords.to_json('dalla_headwords.json', orient='records')
    #df_dalla_entries.to_json('dalla_entries.json', orient='records')
    #df_rae_headwords.to_json('rae_headwords.json', orient='records')
    #df_rae_entries.to_json('rae_entries.json', orient='records')



    df_dalla = pd.DataFrame({
        'ID': [i for i in range(len(headword_list))],
        'Masculine': [tuple[0] for tuple in headword_list],
        'Femenine': [tuple[1] for tuple in headword_list],
        'Definitions': [word_descriptions for word_descriptions in description_list]
    })

    df_rae = pd.DataFrame({
        'ID': [i for i in range(len(final_document_headwords))],
        'Masculine': [tuple[0] for tuple in final_document_headwords],
        'Femenine': [tuple[1] for tuple in final_document_headwords],
        'Definitions': [word_descriptions for word_descriptions in final_document_entries]
    })

    df_dalla.to_json('../data/json/dalla.json', orient='records')
    df_rae.to_json('../data/json/rae.json', orient='records')
    

if __name__=="__main__":
    main()