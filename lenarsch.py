import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words, remove stopwords and punctuation
    tokenized_sentences = []
    stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now" ])
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        tokenized_sentences.append(filtered_words)

    return tokenized_sentences


def calculate_sentence_scores(tokenized_sentences):
    # Calculate term frequency for each word in the text
    word_frequency = FreqDist()
    for sentence in tokenized_sentences:
        for word in sentence:
            word_frequency[word] += 1

    # Calculate the score for each sentence based on word frequency
    sentence_scores = {}
    for sentence in tokenized_sentences:
        score = 0
        for word in sentence:
            score += word_frequency[word]
        sentence_scores[' '.join(sentence)] = score

    return sentence_scores


def calculate_word_scores(tokenized_sentences):
    # Calculate bigram scores based on Pointwise Mutual Information (PMI)
    bigram_finder = BigramCollocationFinder.from_documents(tokenized_sentences)
    bigram_scores = {bigram: BigramAssocMeasures.pmi(*bigram) for bigram, _ in bigram_finder.score_ngrams(BigramAssocMeasures.pmi)}

    # Calculate word scores using SingleRank formula
    word_scores = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            if word in bigram_scores:
                if word in word_scores:
                    word_scores[word] += bigram_scores[word]
                else:
                    word_scores[word] = bigram_scores[word]

    return word_scores


def single_rank(string, text):
    # Preprocess the text and tokenize into sentences and words
    tokenized_sentences = preprocess_text(text)

    # Calculate the scores for each sentence and word
    sentence_scores = calculate_sentence_scores(tokenized_sentences)
    word_scores = calculate_word_scores(tokenized_sentences)

    # Tokenize the string and calculate the score based on word occurrences
    string_words = word_tokenize(string.lower())
    string_score = sum(word_scores.get(word, 0) for word in string_words)

    return string_score


# Example usage
string = "Peter plays football"
text = "Peter plays Football. He is hungry. Magarete plays football too."

score = single_rank(string, text)
print(f"The SingleRank score for '{string}' is: {score}")
