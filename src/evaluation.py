import utils

evaluation_file = '../data/test/test.json'
edit_distance_file = '../data/alignments/edit_distance_alignment.json'
embedding_similarity_file = '../data/alignments/embedding_distance_alignment.json'
llm_file = '../data/alignments/llm_alignment.json'

"""Calculate accuracy"""
def calculate_accuracy(alignments, evaluation_set):
    total_alignments = len(evaluation_set)
    correct_alignments = 0
    correct_alignments = sum(1 for key in evaluation_set if evaluation_set[key] in alignments[key])
    accuracy = correct_alignments/total_alignments
    return accuracy

def main():
    # Get evaluation set and alignments
    evaluation_set = utils.load_json_file(evaluation_file)
    edit_distance_alignment = utils.load_json_file(edit_distance_file)
    embedding_similarity_alignment = utils.load_json_file(embedding_similarity_file)
    llm_alignment = utils.load_json_file(llm_file)

    print(f'Edit distance accuracy: {calculate_accuracy(edit_distance_alignment, evaluation_set):.2f}')
    print(f'Embedding similarity accuracy: {calculate_accuracy(embedding_similarity_alignment, evaluation_set):.2f}')
    print(f'Large language model accuracy: {calculate_accuracy(llm_alignment, evaluation_set):.2f}')

if __name__=='__main__':
    main()