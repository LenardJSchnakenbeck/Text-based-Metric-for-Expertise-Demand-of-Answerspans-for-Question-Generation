import json
import nltk
#nltk.download('averaged_perceptron_tagger')

def candidate_chunking(text, candidate_pos=["CD", "JJ", "NN", "NNP", "NNS", "ADJ", "VBG", "RBS", "PRP"]):
                                                                                #follwing, most
    # TODO: tokenization: wenn ' im Word: doesn't wird zu ('doesn', 'VBZ'), ('’', 'JJ'), ('t', 'NN')
    # TODO: chunking: wenn Date + NN -> nicht verbinden (2014 cars)
    # TODO: chunking: wenn iwas, dann chunk das zusammen, nicht weniger, aber evtl. mehr

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    candidates = []
    current_group = []
    for i in range(len(tagged)):
        if tagged[i][1] in candidate_pos or tagged[i][1] is True:
            current_group += [tagged[i][0]]
        else:
            if current_group:
                candidates += [current_group]
                current_group = []
            candidates += [tagged[i][0]]

    return candidates, tokens, tagged


def chunking_texts(documents):
    labeled_documents = []
    for document in documents:
        candidate_chunks, tokens, _ = candidate_chunking(document["text"])
        document['candidates'] = candidate_chunks
        document['tokens'] = tokens
        labeled_documents.append(document)
    return labeled_documents


def write_json_labeled_documents(source_documents_path, labeled_documents_path):
    text_labeled = open(source_documents_path, encoding="utf-8")
    documents = json.load(text_labeled)
    with open(labeled_documents_path, 'w') as f:
        json.dump(chunking_texts(documents), f)

if __name__ == "__main__":
    write_json_labeled_documents('wikipedia_texts.json','labeled_wikipedia_texts.json')





