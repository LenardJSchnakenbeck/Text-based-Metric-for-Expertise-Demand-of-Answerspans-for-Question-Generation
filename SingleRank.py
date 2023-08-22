import json
import networkx as nx
import pickle


def word_scoring(chunked_candidates, window = 10):
    words = []
    #flatten chunked_candidates
    for word_or_list in chunked_candidates:
        iscandidate = True if isinstance(word_or_list, list) else False
        if iscandidate:
            words += [(word,True) for word in word_or_list]
        else:
            words += [(word_or_list, False)]

    #create Graph
    graph = nx.Graph()
    graph.add_nodes_from([word for word, iscandidate in words if iscandidate])  # words = [(word,True), (word,False)]

    # add edges to the graph
    for i, (node1, is_in_graph1) in enumerate(words):

        # speed up things
        if not is_in_graph1:
            continue

        for j in range(i + 1, min(i + window, len(words))):
            node2, is_in_graph2 = words[j]
            if is_in_graph2 and node1 != node2:
                graph.add_edge(node1, node2)

    scores = nx.pagerank(graph,
                    alpha=0.85,
                    tol=0.0001,
                    weight='weight')
    return scores


# Candidate score = sum(word scores)
def candidate_scoring(word_scores, candidates, normalized=False):
    scores_dict = {}
    scores_array = []
    #candidates = [candidate for candidate, iscandidate in chunked_candidates if iscandidate]
    for index, k in enumerate(candidates):
        if isinstance(k, list):
            tokens = k #nltk.word_tokenize(k)
            scores_dict[str(k)] = sum([word_scores[t] for t in tokens])
            scores_array += [sum([word_scores[t] for t in tokens])]
            if normalized:
                scores_dict[str(k)] /= len(tokens)

    # Extract values from the dictionary in the same order as the keys
    dictionary_values = [scores_dict[key] for key in scores_dict]
    # Check if the floats in the list match the corresponding values in the dictionary
    in_same_order = all(a == b for a, b in zip(scores_array, dictionary_values))
    if not in_same_order: raise ValueError("the order was lost when dictionary was converted to list")

    return scores_array

def calculate_and_write_pickle_singlerank_scores(labeled_documents_path, singlerank_scores_path):
    documents_json = open(labeled_documents_path)
    documents = json.load(documents_json)
    #results = []
    for document in documents:
        candidates = document["candidates"]
        document['singlerank_scores'] = candidate_scoring(word_scoring(candidates), candidates)

    with open(singlerank_scores_path, 'wb') as pkl:
        pickle.dump(documents, pkl)

def load_singlerank_scores(singlerank_scores_path):
    with open(singlerank_scores_path, 'rb') as pkl:
        return pickle.load(pkl)


if __name__ == "main":
    labeled_documents_path = 'preprocessing/labeled_documents.json'
    singlerank_scores_path = 'similarity/singlerank_scores.pickle'
    calculate_and_write_pickle_singlerank_scores(labeled_documents_path, singlerank_scores_path)
    singlerank_documents = load_singlerank_scores(singlerank_scores_path)

