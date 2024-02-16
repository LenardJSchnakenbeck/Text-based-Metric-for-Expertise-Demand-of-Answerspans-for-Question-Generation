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
    if pos.startswith('J'):
        return "a"
    elif pos.startswith('V'):
        return "v"
    elif pos.startswith('N'):
        return "n"
    elif pos.startswith('R'):
        return "r"
    else:
        return ''


def get_main_word(candidate, pos_list):
    if len(candidate) > 1 and "NN" in "".join(pos_list):
        indexes = [index for index, pos in enumerate(pos_list) if "NN" in pos]
        main_word = candidate[indexes[-1]]
        pos = "NN"
    elif isinstance(candidate, list):
        main_word = candidate[-1]
        pos = pos_list[-1]
    else:
        raise ValueError("get_main_word: candidate is no list")
    return main_word, pos


def write_json_labeled_documents(source_documents_path, labeled_documents_path):
    text_labeled = open(source_documents_path, encoding="utf-8")
    documents = json.load(text_labeled)
    with open(labeled_documents_path, 'w') as f:
        json.dump(chunking_texts(documents), f)


def load_json_labeled_documents(labeled_documents_path):
    with open(labeled_documents_path, 'rb') as f:
        return json.load(f)

def write_study_texts(labeled_documents_path):
    texts = [{"title": "Earthworms",
      "text": "Earthworms have the ability to regenerate lost segments, but this ability varies between species and depends on the extent of the damage. Stephenson (1930) devoted a chapter of his monograph to this topic, while C.E. Gates (1972) spent 20 years studying regeneration in a variety of species, but „because little interest was shown“, Gates published only a few of his findings that, nevertheless, show it is theoretically possible to grow two whole worms from a bisected specimen in certain species.",
      "candidates": [['decease'],['Earthworms'],'have', ['the', 'ability', 'to', 'regenerate', 'lost', 'segments'], ',', 'but', 'this', 'ability', 'varies', 'between', ['species'], 'and', 'depends', 'on', ['the', 'extent', 'of', 'the', 'damage'], '.', ['Stephenson'], '(', '1930', ')', 'devoted', ['a', 'chapter', 'of', 'his', 'monograph'], 'to', 'this', 'topic', ',', 'while', ['C.E.Gates'], '(', '1972', ')', 'spent', ['20', 'years'], 'studying', ['regeneration', 'in', 'a', 'variety', 'of', 'species'], ',', 'but', '„', 'because', 'little', 'interest', 'was', 'shown', '“', ',', ['Gates'],  'published', 'only', 'a', 'few', 'of', 'his', 'findings', 'that', ',', 'nevertheless', ',', 'show', 'it', 'is', 'theoretically', 'possible', 'to', 'grow', ['two', 'whole', 'worms'], 'from', ['a', 'bisected', 'specimen'], 'in', ['certain', 'species'], '.'],
      "tokens": ['decease','Earthworms', 'have', 'the', 'ability', 'to', 'regenerate', 'lost', 'segments', ',', 'but', 'this', 'ability', 'varies', 'between', 'species', 'and', 'depends', 'on', 'the', 'extent', 'of', 'the', 'damage', '.', 'Stephenson', '(', '1930', ')', 'devoted', 'a', 'chapter', 'of', 'his', 'monograph', 'to', 'this', 'topic', ',', 'while', 'C.E', '.', 'Gates', '(', '1972', ')', 'spent', '20', 'years', 'studying', 'regeneration', 'in', 'a', 'variety', 'of', 'species', ',', 'but', '„', 'because', 'little', 'interest', 'was', 'shown', '“', ',', 'Gates', 'published', 'only', 'a', 'few', 'of', 'his', 'findings', 'that', ',', 'nevertheless', ',', 'show', 'it', 'is', 'theoretically', 'possible', 'to', 'grow', 'two', 'whole', 'worms', 'from', 'a', 'bisected', 'specimen', 'in', 'certain', 'species', '.'],
      "candidates_pos": ['NNS','NNS', 'VBP', 'DT', 'NN', 'TO', 'VB', 'VBN', 'NNS', ',', 'CC', 'DT', 'NN', 'VBZ', 'IN', 'NNS', 'CC', 'VBZ', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', '.', 'NNP', '(', 'CD', ')', 'VBD', 'DT', 'NN', 'IN', 'PRP$', 'NN', 'TO', 'DT', 'NN', ',', 'IN', 'NNP', '.', 'NNP', '(', 'CD', ')', 'VBD', 'CD', 'NNS', 'VBG', 'NN', 'IN', 'DT', 'NN', 'IN', 'NNS', ',', 'CC', 'NNP', 'RB', 'JJ', 'NN', 'VBD', 'VBN', 'NNP', ',', 'NNP', 'VBD', 'RB', 'DT', 'JJ', 'IN', 'PRP$', 'NNS', 'IN', ',', 'RB', ',', 'VB', 'PRP', 'VBZ', 'RB', 'JJ', 'TO', 'VB', 'CD', 'JJ', 'NNS', 'IN', 'DT', 'JJ', 'NNS', 'IN', 'JJ', 'NNS', '.'],
      "tokens_pos": [('decease','NNS'),('Earthworms', 'NNS'), ('have', 'VBP'), ('the', 'DT'), ('ability', 'NN'), ('to', 'TO'), ('regenerate', 'VB'), ('lost', 'VBN'), ('segments', 'NNS'), (',', ','), ('but', 'CC'), ('this', 'DT'), ('ability', 'NN'), ('varies', 'VBZ'), ('between', 'IN'), ('species', 'NNS'), ('and', 'CC'), ('depends', 'VBZ'), ('on', 'IN'), ('the', 'DT'), ('extent', 'NN'), ('of', 'IN'), ('the', 'DT'), ('damage', 'NN'), ('.', '.'), ('Stephenson', 'NNP'), ('(', '('), ('1930', 'CD'), (')', ')'), ('devoted', 'VBD'), ('a', 'DT'), ('chapter', 'NN'), ('of', 'IN'), ('his', 'PRP$'), ('monograph', 'NN'), ('to', 'TO'), ('this', 'DT'), ('topic', 'NN'), (',', ','), ('while', 'IN'), ('C.E', 'NNP'), ('.', '.'), ('Gates', 'NNP'), ('(', '('), ('1972', 'CD'), (')', ')'), ('spent', 'VBD'), ('20', 'CD'), ('years', 'NNS'), ('studying', 'VBG'), ('regeneration', 'NN'), ('in', 'IN'), ('a', 'DT'), ('variety', 'NN'), ('of', 'IN'), ('species', 'NNS'), (',', ','), ('but', 'CC'), ('„', 'NNP'), ('because', 'RB'), ('little', 'JJ'), ('interest', 'NN'), ('was', 'VBD'), ('shown', 'VBN'), ('“', 'NNP'), (',', ','), ('Gates', 'NNP'), ('published', 'VBD'), ('only', 'RB'), ('a', 'DT'), ('few', 'JJ'), ('of', 'IN'), ('his', 'PRP$'), ('findings', 'NNS'), ('that', 'IN'), (',', ','), ('nevertheless', 'RB'), (',', ','), ('show', 'VB'), ('it', 'PRP'), ('is', 'VBZ'), ('theoretically', 'RB'), ('possible', 'JJ'), ('to', 'TO'), ('grow', 'VB'), ('two', 'CD'), ('whole', 'JJ'), ('worms', 'NNS'), ('from', 'IN'), ('a', 'DT'), ('bisected', 'JJ'), ('specimen', 'NNS'), ('in', 'IN'), ('certain', 'JJ'), ('species', 'NNS'), ('.', '.')]
         }]
    #answerspans = ["Earthworms", "the ability to regenerate lost segments", "species", "the extent of the damage",
    #    "Stephenson", "a chapter of his monograph", "C.E.Gates", "20 years", "regeneration in a variety of species",
    #    "Gates", "two whole worms", "a bisected specimen", "certain species"]
    with open(labeled_documents_path, 'w') as f:
        json.dump(texts, f)

if __name__ == "__main__":
    write_json_labeled_documents('wikipedia_texts.json','labeled_wikipedia_texts.json')





