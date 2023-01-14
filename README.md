# IR-Project

# Description
This project contains functions for a search engine that can tokenize text, calculate the cosine similarity between a query and document bodies, search for documents based on their titles and anchors, and merge scores from different sources to return a list of relevant documents.

# Code Structure
The project contains 4 helper functions in search_backend.py:

tokenize (text, use_stemming = False): tokenize the input text and returns a list of tokens. It removes stopwords and applies stemming (optional)
cossim_body (query_tok, index_body, DL, idf, W_ij, topN, blob_name): returns the top N documents that are most similar to the query based on the cosine similarity between the query and the document bodies.
search_title_helper (query_tok, index_title, blob_name): returns a list of documents sorted by the number of shared terms between the query and the document title.
search_anchor_helper (query_tok, index_anchor): returns a list of documents sorted by the number of shared terms between the query and the document anchor.
merge_scores (q_len, title_scores, body_scores, pr, title_weight=0.5, body_weight=0.5, topN=100): merges the scores from different sources, normalizes the page rank score, and returns the top N documents.

The project also contains 5 main functions in search_frontend.py:

search (): Returns up to a 100 of the best search results for the query. It uses the previous defined functions to perform the search.
search_body (): Returns up to a 100 search results for the query using TFIDF AND COSINE SIMILARITY OF THE BODY OF ARTICLES ONLY.
search_title (): Returns ALL (not just top 100) search results that contain A QUERY WORD IN THE TITLE of articles, ordered in descending order of the NUMBER OF QUERY WORDS that appear in the title.
get_pagerank (): function takes a list of Wikipedia article IDs and returns the PageRank values for each article
get_pageview (): function takes a list of Wikipedia article IDs and returns the number of page views for each article in August 2021.


•	index_body, index_title and index_anchor objects that provide the posting lists for each term and document
•	DL (document length) dictionary
•	idf (inverse document frequency) dictionary
•	W_ij (normal of the document) dictionary
•	pr (page rank) dictionary
•	titles dictionary that provide the title of each document

