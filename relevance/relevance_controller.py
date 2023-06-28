import nltk

def candidate_chunking(documents, candidates):
    text = "Hello my name is your mom"
    tokens = nltk.word_tokenize(text)
    candidates = ["my name is", "your mom"]


    chunks = []
    #tokens = documents["tokenized_text"]
    candidate_index = 0
    max_candidate_length = max([len(nltk.word_tokenize(can)) for can in candidates]) # +3,bc .split() != tokenization
    #Iterate through all tokens
    i = 0
    while i < len(tokens):
        if candidate_index < len(candidates):
            candidate = candidates[candidate_index]
            if tokens[i] in candidate:
                c = 0
                for j in range(len(nltk.word_tokenize(candidate))+1):
                    if nltk.word_tokenize(candidate) == tokens[i: min(i+j, len(tokens))]:
                        c = tokens[i:min(i+j, len(tokens))]
                if c:
                    chunks += [[c, True]]
                    candidate_index += 1
                    i += len(c)-1
            else:
                chunks += [[tokens[i], False]]
        else:
            chunks += [[tokens[i], False]]
        i += 1
    return chunks

print(candidate_chunking(None, None))
#Pepe