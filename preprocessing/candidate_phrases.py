import json
import nltk
#nltk.download('averaged_perceptron_tagger')


def load_wikipedia_and_create_json(path_to_raw_wikipedia_texts, source_documents_path, number_of_lines=10000):
    #load
    with open(path_to_raw_wikipedia_texts, "r") as f:
        if number_of_lines:
            texts = f.readlines(number_of_lines)
        else:
            texts = f.readlines()
    texts.pop(-1)
    #refactor
    texts_dicts = []
    for text in texts:
        separator = text.index(" ||| ")
        title = text[:separator]
        texts_dicts += [{"title": title, "text": text[separator + 5:]}]
    #save
    with open(source_documents_path, "w") as f:
        json.dump(texts_dicts, f)


def candidate_chunking(text, candidate_pos=["CD", "JJ", "NN", "NNP", "NNS", "ADJ", "VBG", "RBS", "PRP"]):
                                                                                #follwing, most
    # TODO: tokenization: wenn ' im Word: doesn't wird zu ('doesn', 'VBZ'), ('’', 'JJ'), ('t', 'NN')
    # TODO: chunking: wenn Date + NN -> nicht verbinden (2014 cars)
    # TODO: chunking: wenn iwas, dann chunk das zusammen, nicht weniger, aber evtl. mehr

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tagged_candidates = []

    candidates = []
    current_group = []
    for i in range(len(tagged)):
        if tagged[i][1] in candidate_pos or tagged[i][1] is True:
            current_group += [tagged[i]]
        else:
            if current_group:
                candidates += [[i[0] for i in current_group]] #word
                tagged_candidates += [i[1] for i in current_group] #pos
                current_group = []
            candidates += [tagged[i][0]]
            tagged_candidates += [tagged[i][1]]

    return candidates, tokens, tagged_candidates, tagged


def chunking_texts(documents):
    labeled_documents = []
    for document in documents:
        candidate_chunks, tokens, tagged_candidates, tagged_tokens = candidate_chunking(document["text"])
        document['candidates'] = candidate_chunks
        document['tokens'] = tokens
        document['candidates_pos'] = tagged_candidates
        document['tokens_pos'] = tagged_tokens
        labeled_documents.append(document)
    return labeled_documents


def map_pos_tokenizer_to_lemmatizer(pos):
    pos = pos[0].lower()
    if pos not in ["n", "v", "a", "r", "s"]:
        pos = "n"
    return pos


def get_main_word(candidate, pos):
    print("get_main_word", candidate, pos)
    if len(candidate) > 1 and "n" in pos:
        indexes = [index for index, value in enumerate(pos) if value == "n"]
        candidate = candidate[indexes[-1]]
        pos = "n"
    elif isinstance(candidate, list):
        candidate = candidate[-1]
        pos = pos[-1]
    else:
        raise ValueError("get_main_word: candidate is no list")
    return candidate, pos


def write_json_labeled_documents(source_documents_path, labeled_documents_path):
    text_labeled = open(source_documents_path, encoding="utf-8")
    documents = json.load(text_labeled)
    with open(labeled_documents_path, 'w') as f:
        json.dump(chunking_texts(documents), f)

if __name__ == "__main__":
    write_json_labeled_documents('wikipedia_texts.json','labeled_wikipedia_texts.json')





