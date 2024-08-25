from techniques import get_words_by_edit_distance, get_words_by_embeddings_distance, get_words_by_llm, flatten_words_and_definitions

api_key = 'sk-None-YeupLbSg61AXwdTDdUOJT3BlbkFJWL3VWN23Cqj5RJRfRokR'
evaluation_words = {'abrochar':'abrochar', 'bocarón':'boquerón', 'corderu': 'cordero', 'maña':'destreza', 'edredón':'edredón',
                    'faxu':'fajo', 'xeneración':'generación', 'foguera':'hoguera', 'infrarroxu':'infrarrojo', 'xíriga':'jerga',
                    'quilogramu':'kilogramo', 'llaberínticu':'laberíntico', 'cotra':'mugre', 'oveya':'oveja', 'neñina':'pupila',
                    'quexa':'queja', 'rayu':'rayo', 'sarapicu':'sarampión', 'títulu':'título', 'solombra':'umbría',
                    'vocal':'vocal', 'xenofobia':'xenofobia', 'yacimientu':'yacimiento', 'zancu':'zanco', 'anticuáu':'anticuado',
                    'brúxula':'brújula', 'calzáu':'calzado', 'esbancar':'desbancar', 'enxendrar':'engendrar', 'altimora':'frambuesa',
                    'xeometría':'geometría', 'filera':'hilera', 'innatu':'innato', 'xabalín':'jabalí', 'quioscu':'kiosco',
                    'llinu':'lino', '':'', '':'', '':'', '':'',
                    '':'', '':'', '':'', '':'', '':'',
                    '':'', '':'', '':'', '':'', '':'',


                    'perceición':'percepción', 'tornar':'volver',
                    '':'', '':'', '':'', '':'', '':''}

def main():
    dictionary = flatten_words_and_definitions()
    words_to_search = ['bonitu', 'abusivu']
    print(get_words_by_edit_distance(words_to_search))
    print(get_words_by_embeddings_distance(['sust. Primera lletra [del abecedariu] que se representa por "a".', 'ax. Que ye un abusu.'], dictionary))
    exit()
    words_to_search = ['a', 'abusivu']
    definitions_to_search = ['sust. Primera lletra [del abecedariu] que se representa por "a".', 'ax. Que ye un abusu.']
    print(get_words_by_llm(words_to_search, definitions_to_search, api_key))
    

if __name__=='__main__':
    main()