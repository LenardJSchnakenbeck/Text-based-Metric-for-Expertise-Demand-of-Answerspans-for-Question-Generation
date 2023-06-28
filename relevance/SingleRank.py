import nltk
import pke
import networkx as nx
# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}
# 1. create a SingleRank extractor.
extractor = pke.unsupervised.SingleRank()
# 2. load the content of the document.
extractor.load_document(input='Pond herons (Ardeola) are herons from the University of Duisburg, typically 40–50 cm (16–20 in) long with an 80–100 cm (30–40 in) wingspan. Most breed in the tropical Old World, but the migratory squacco heron occurs in southern Europe and the Middle East and winters in Africa. The scientific name comes from Latin ardeola, a small heron (ardea).',
                        language='en',
                        normalization=None)

# 3. select the longest sequences of nouns and adjectives as candidates.
extractor.candidate_selection(pos=pos)
# 4. weight the candidates using the sum of their word's scores that are
#    computed using random walk. In the graph, nodes are words of
#    certain part-of-speech (nouns and adjectives) that are connected if
#    they occur in a window of 10 words.

graph = extractor.graph
words = extractor.candidates
sents = extractor.sentences
weigths = extractor.weights

extractor.candidate_weighting(window=10,
                              pos=pos)
# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

class Reader(object):
    """Reader default class."""

    def read(self, path):
        raise NotImplementedError

class RawTextReader(Reader):
    """Reader for raw text."""

    def __init__(self, language=None):
        """Constructor for RawTextReader.

        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:
            self.language = 'en'

        if len(self.language) != 2:
            raise ValueError('`language` is \'{}\', but should be an iso2 language code (\'en\' instead of \'english\')'.format(self.language))

    def read(self, text, spacy_model=None):
        """Read the input file and use spacy to pre-process.

        Spacy model selection: By default this function will load the spacy
        model that is closest to the `language` parameter ('fr' language will
        load the spacy model linked to 'fr' or any 'fr_core_web_*' available
        model). In order to select the model that will be used please provide a
        preloaded model via the `spacy_model` parameter, or link the model you
        wish to use to the corresponding language code
        `python3 -m spacy link spacy_model lang_code`.

        Args:
            text (str): raw text to pre-process.
            spacy_model (model): an already loaded spacy model.
        """

        nlp = spacy_model

        if nlp is None:

            # list installed models
            installed_models = [m for m in spacy.util.get_installed_models() if m[:2] == self.language]

            # select first model for the language
            if len(installed_models):
                nlp = spacy.load(installed_models[0], disable=['ner', 'textcat', 'parser'])

            # stop execution is no model is available
            else:
                excp_msg = 'No downloaded spacy model for \'{}\' language.'.format(self.language)
                excp_msg += '\nA list of downloadable spacy models is available at https://spacy.io/models.'
                excp_msg += '\nAlternatively, preprocess your document as a list of sentence tuple (word, pos), such as:'
                excp_msg += "\n\t[[('The', 'DET'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('.', 'PUNCT')]]"
                raise Exception(excp_msg)

            # add the sentence splitter
            nlp.add_pipe('sentencizer')

        # Fix for non splitting words with hyphens with spacy taken from
        # https://spacy.io/usage/linguistic-features#native-tokenizer-additions
        nlp.tokenizer.infix_finditer = infix_re.finditer

        # process the document
        spacy_doc = nlp(text)

        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append(Sentence(
                words=[token.text for token in sentence],
                pos=[token.pos_ or token.tag_ for token in sentence],
                meta={
                    "lemmas": [token.lemma_ for token in sentence],
                    "char_offsets": [(token.idx, token.idx + len(token.text))
                                     for token in sentence]
                }
            ))
        return sentences


#if pos is None:
pos = {'NOUN', 'PROPN', 'ADJ'}

window = 2
spacy_model = None
language = "en"
input = 'Pond herons (Ardeola) are herons from the University of Duisburg, typically 40–50 cm (16–20 in) long with an 80–100 cm (30–40 in) wingspan. Most breed in the tropical Old World, but the migratory squacco heron occurs in southern Europe and the Middle East and winters in Africa. The scientific name comes from Latin ardeola, a small heron (ardea).'
parser = RawTextReader(language=language)
sentences = parser.read(text=input, spacy_model=spacy_model)

# flatten document as a sequence of (word, pass_syntactic_filter) tuples
text = [(word, sentence.pos[i] in pos) for sentence in sentences
        for i, word in enumerate(sentence.stems)]

candidates = document["noun_phrases"]
normalized = True

def word_scoring():
    graph = nx.Graph()
    graph.add_nodes_from([word for word, valid in text if valid]) # text = [(word,True), (word,False)]

    # add edges to the graph
    for i, (node1, is_in_graph1) in enumerate(text):

        # speed up things
        if not is_in_graph1:
            continue

        for j in range(i + 1, min(i + window, len(text))):
            node2, is_in_graph2 = text[j]
            if is_in_graph2 and node1 != node2:
                graph.add_edge(node1, node2)

    scores = nx.pagerank(graph,
                    alpha=0.85,
                    tol=0.0001,
                    weight='weight')
    return scores

#J: Candidate score = avg(word scores)
# loop through the candidates
def candidate_scoring():
    scores = []
    for k in candidates:
        tokens = nltk.word_tokenize(k)
        scores[k] = sum([w[t] for t in tokens]) #
        if normalized:
            scores[k] /= len(tokens)
        # use position to break ties
        #???
        scores[k] += (candidates[k].offsets[0] * 1e-8)
    return scores
