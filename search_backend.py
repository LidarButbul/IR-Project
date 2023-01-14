from collections import Counter, OrderedDict
import math
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')

stemmer = PorterStemmer()
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text, use_stemming = False):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [token for token in tokens if (token not in all_stopwords)]
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def cossim_body(query_tok, index_body, DL, idf, W_ij, topN, blob_name):
    candidates = {}
    query_tfidf_dict = {}  #key: term, value: tf-idf-query
    w_iq_denom = 0
    query_tf = Counter(query_tok)
    for term in query_tok:
        query_tf_norm = query_tf[term] / len(query_tok)
        idf_val = idf.get(term, 0)
        w_iq = query_tf_norm * idf_val
        query_tfidf_dict[term] = w_iq
        w_iq_denom += w_iq ** 2
    w_iq_denom = math.sqrt(w_iq_denom)
    for term in query_tok:
        posting = index_body.read_posting_list(term, blob_name)
        for doc_id, tf in posting:
            tfidf = (tf / DL[doc_id]) * idf.get(term, 0)
            candidates[doc_id] = candidates.get(doc_id, 0) + ((tfidf) * query_tfidf_dict[term]) / ((W_ij[doc_id]) * w_iq_denom)
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:topN]

def search_title_helper(query_tok, index_title, blob_name):
    docs = {}  #key: doc_id, value: number of shared terms(query and title)
    for term in query_tok:
        posting = index_title.read_posting_list(term, blob_name)
        for doc_id, num in posting:
            docs[doc_id] = docs.get(doc_id, 0) + 1
    return sorted(docs.items(), key=lambda x: x[1], reverse=True)

def search_anchor_helper(query_tok, index_anchor):
    docs = {}  #key: doc_id, value: number of shared terms (query and anchor)
    for term in query_tok:
        posting = index_anchor.read_posting_list(term, 'postings_anchor')
        for doc_id, num in posting:
            docs[doc_id] = docs.get(doc_id, 0) + 1
    return sorted(docs.items(), key=lambda x: x[1], reverse=True)

def merge_scores(q_len, title_scores,body_scores, pr, title_weight=0.5,body_weight=0.5,topN=100):
    pr_max = max(pr.values())
    pr_min = min(pr.values())
    title_scores_dict = dict(title_scores)
    body_scores_dict = dict(body_scores)
    title_docs = list(title_scores_dict.keys())
    body_docs = list(body_scores_dict.keys())
    all_docs = set(title_docs + body_docs)
    docs = {}
    for doc in all_docs:
        if doc in title_scores_dict.keys():
            new_score = title_weight * (title_scores_dict[doc] / q_len)
            if doc in body_scores_dict.keys():
                new_score += body_weight * body_scores_dict[doc]
        else:
            new_score = body_weight * body_scores_dict[doc]
        if doc in pr.keys():
            pr_norm = (pr[doc] - pr_min) / (pr_max - pr_min)
            docs[doc] = 0.4 * new_score + 0.6 * pr_norm
        else:
            docs[doc] = 0.4 * new_score
    return sorted(docs.items(), key=lambda x: x[1], reverse=True)[:topN]







