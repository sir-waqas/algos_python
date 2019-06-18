from nltk.tokenize import word_tokenize
import gensim
# print(dir(gensim))
raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
                 "I am the barber who cuts everyone's hair who doesn't cut their own.",
                 "Legend has it that the mind is a mad monkey.",
                 "I make my own fun."]
print("Number of documents:", len(raw_documents))

# NLTK to tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in raw_documents]
# print(gen_docs)

# Words to numeric dictionary
dictionary = gensim.corpora.Dictionary(gen_docs)
print('Number of words in this deictionary: ', len(dictionary))
# print(dictionary[5])
# print(dictionary.token2id['road'])

# Let's now create the bag of words representing count for each word's occurance
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

# Now the TF-IDF(TermFrequency--InverseDocumentFrequency) model from this corpus
tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

# s = 0
# for i in corpus:
#     s += len(i)
# print(s)
