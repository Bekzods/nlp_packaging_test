import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en')
import pandas as pd

def lemmatize(doc):
    #doc = nlp(doc)
    return[
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_space
        and (token.text == "US"or not token.lower_ in STOP_WORDS)
        and not token.tag_ == "POS" or token.text == "'s"
     ]

def tf(s, doc):
    # s and doc terms which we calculate if

    lem = lemmatize(doc)

    return lem.count(''.join(lemmatize(nlp(s))))




def idf(s, docs):
    # s calculation IDF
    # docs list of spacy
    doc_counter = 0

    for i in docs:
        lem = lemmatize(i)
        if lem.count(''.join(lemmatize(nlp(s)))) > 0:
            doc_counter += 1
    if doc_counter > 0:
        return 1 / doc_counter
    else:
        return 0


def tf_idf(s, doc, docs):
    sim = [doc.similarity(i) for i in docs]

    if 1.0 in sim:
        term_freq = tf(s, doc)
        inv_doc_freq = idf(s, docs)
        return term_freq * inv_doc_freq
    else:
        raise ValueError('The doc is not in the list ))')



def all_lemmas(docs):
    #docs over the scacy
    lemmas = set()
    for doc in docs:
        lemmas |= set(lemmatize(doc))
    return lemmas


def tf_idf_doc(doc, docs):
    dic = {}
    lemmas = all_lemmas(docs)
    for lemma in lemmas:
        dic[lemma] = tf_idf(lemma, doc, docs)

    return dic


def tf_idf_scores(docs):
    _list = []
    index = [i for i in range(len(docs))]
    for doc in docs:
        _list.append(tf_idf_doc(doc, docs))
    df = pd.DataFrame(data=_list, index=index)
    return df







