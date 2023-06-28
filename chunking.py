
import spacy
import nltk
nlp = spacy.load("en_core_web_sm")

def noun_phrase_chunking(text):
    chunked_text = nlp(text)
    chunks = [nltk.word_tokenize(chunk.text) for chunk in chunked_text.noun_chunks]
    return chunks

def create_chunk_array(text, chunks):
    tokenized_text = nltk.word_tokenize(text)
    chunk_array = []

    j = 0
    for i in range(len(tokenized_text)):
        if j < len(chunks) and tokenized_text[i] in chunks[j]:
            chunk_array += [j]
            if tokenized_text[i] == chunks[j][-1]:
                j += 1
        else:
            chunk_array += [-1]
    df = pd.DataFrame()
    df["tokens"] = tokenized_text
    df["chunks"] = chunk_array
    return df


a = []
for i in range(len(articles)):
    article = articles.iloc[i]["text"]
    title = articles.iloc[i]["title"]
    a += [title, create_chunk_array(article, noun_phrase_chunking(article))]


def relevance_score(a):
    for i in range(len(a)):
        df = a[i][1]
        n_chunks = max(df["chunks"])


    return

