from techniques import get_words_by_edit_distance, get_words_by_embeddings_distance, get_words_by_llm, flatten_words_and_definitions

api_key = 'sk-None-YeupLbSg61AXwdTDdUOJT3BlbkFJWL3VWN23Cqj5RJRfRokR'
evaluation_set = {'abrochar':'abrochar', 'bocarón':'boquerón', 'corderu': 'cordero', 'maña':'destreza', 'edredón':'edredón',
                    'faxu':'fajo', 'xeneración':'generación', 'foguera':'hoguera', 'infrarroxu':'infrarrojo', 'xíriga':'jerga',
                    'quilogramu':'kilogramo', 'llaberínticu':'laberíntico', 'cotra':'mugre', 'oveya':'oveja', 'neñina':'pupila',
                    'quexa':'queja', 'rayu':'rayo', 'sarapicu':'sarampión', 'títulu':'título', 'solombra':'umbría',
                    'vocal':'vocal', 'xenofobia':'xenofobia', 'yacimientu':'yacimiento', 'zancu':'zanco', 'anticuáu':'anticuado',
                    'brúxula':'brújula', 'calzáu':'calzado', 'esbancar':'desbancar', 'enxendrar':'engendrar', 'altimora':'frambuesa',
                    'xeometría':'geometría', 'filera':'hilera', 'innatu':'innato', 'xabalín':'jabalí', 'quioscu':'kiosco',
                    'llinu':'lino', 'matrimoniu':'matrimonio', 'nabu':'nabo', 'arzuelu':'orzuelo', 'pergamín':'pergamino',
                    'quirúrxicu':'quirúrgico', 'retrayer':'retraer', 'sábadu':'sábado', 'tuertu':'tuerto', 'usté':'usted',
                    'cacía':'vajilla', 'xilófonu':'xilófono', 'xunca':'yunque', 'manzorgu':'zurdo', 'miruéndanu':'arándano',
                    'bordáu':'bordado', 'costáu':'costado', 'didal':'dedal', 'elipse':'elipse', 'frenar':'frenar',
                    'xéneru':'género', 'faba':'haba', 'impar':'impar', 'enxamás':'jamás', 'quilovatiu':'kilovatio',
                    'llabia':'labia', 'miel':'miel', 'numberación':'numeración', 'ortopédicu':'ortopédico', 'perceición':'percepción',
                    'quexada':'quijada', 'restaurante':'restaurante', 'serviyeta':'servilleta', 'tamién':'también', 'xunión':'unión',
                    'tornar':'volver', 'xilografía':'xilografía', 'yá':'ya', 'zarpa':'zarpa', 'allacrán':'alacrán',
                    'brasileñu':'brasileño', 'cianuru':'cianuro', 'dulda':'duda', 'estilu':'estilo', 'familia':'familia',
                    'ximnasia':'gimnasia', 'güevera':'huevera', 'intransixencia':'intransigencia', 'xuncu':'junco', 'llista':'lista',
                    'mangu':'mango', 'ninfa':'ninfa', 'óvalu':'óvalo', 'poema':'poema', 'quiciás':'quizá',
                    'rastrexar':'rastrear', 'seta':'seta', 'tomillu':'tomillo', 'unidá':'unidad', 'vidrera':'vidriera',
                    'yo':'yo', 'zafiru':'zafiro', 'equí':'aquí', 'bendicir':'bendecir', 'contraición':'contracción'}

def main():
    dictionary = flatten_words_and_definitions()
    x = evaluation_set.keys()
    expected_y = evaluation_set.values()
    returned_y = get_words_by_edit_distance(x)
    correct_alignments = 0
    for returned, expected in zip(returned_y, expected_y):
        if returned == expected:
            correct_alignments += 1
    accuracy = correct_alignments/len(expected_y)
    print(f'Accuracy: {accuracy:.2f}')
    exit()
        
    words_to_search = ['bonitu', 'abusivu']
    print(get_words_by_edit_distance(words_to_search))
    print(get_words_by_embeddings_distance(['sust. Primera lletra [del abecedariu] que se representa por "a".', 'ax. Que ye un abusu.'], dictionary))
    exit()
    words_to_search = ['a', 'abusivu']
    definitions_to_search = ['sust. Primera lletra [del abecedariu] que se representa por "a".', 'ax. Que ye un abusu.']
    print(get_words_by_llm(words_to_search, definitions_to_search, api_key))
    

if __name__=='__main__':
    main()