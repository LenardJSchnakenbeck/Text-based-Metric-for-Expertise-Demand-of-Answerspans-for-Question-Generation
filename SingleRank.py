import nltk
import pke
import networkx as nx
# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}
# 1. create a SingleRank extractor.

#window = 2
#spacy_model = None
#language = "en"
input = 'Pond herons (Ardeola) are herons from the University of Duisburg, typically 40–50 cm (16–20 in) long with an 80–100 cm (30–40 in) wingspan. Most breed in the tropical Old World, but the migratory squacco heron occurs in southern Europe and the Middle East and winters in Africa. The scientific name comes from Latin ardeola, a small heron (ardea).'
#candidates = document["noun_phrases"]

def word_scoring(chunked_candidates, window = 10):
    chunked_candidates = [['Hello', False], [['my', 'name', 'is'], True], [['your', 'mom'], True]]

    words = []
    #flatten chunked_candidates
    for candidate in chunked_candidates:
        word_or_list = candidate[0]
        iscandidate = candidate[1]
        words += [(word_or_list, iscandidate)] if isinstance(word_or_list, str) \
            else list(zip(word_or_list, [iscandidate] * len(word_or_list)))

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
def candidate_scoring(word_scores, chunked_candidates, normalized=False):
    scores = {}
    candidates = [candidate for candidate, iscandidate in chunked_candidates if iscandidate]
    for k in candidates:
        tokens = k #nltk.word_tokenize(k)
        scores[str(k)] = sum([word_scores[t] for t in tokens])
        if normalized:
            scores[str(k)] /= len(tokens)
        #scores[str(k)] += (offset * 1e-8) #???
    return scores
