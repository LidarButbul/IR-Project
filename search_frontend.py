from flask import Flask, request, jsonify
from collections import Counter
import pickle
import pandas as pd
import os
from search_backend import *
import gcsfs

GCSFS = gcsfs.GCSFileSystem()

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        with GCSFS.open('gs://315682872_209505593/index_body.pkl', 'rb') as f:
            self.index_body = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/index_title.pkl', 'rb') as f:
            self.index_title = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/index_anchor.pkl', 'rb') as f:
            self.index_anchor = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/page_rank.pkl', 'rb') as f:
            self.page_rank = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/page_view.pkl', 'rb') as f:
            self.page_view = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/titles.pkl', 'rb') as f:
            self.titles = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/DL.pkl', 'rb') as f:
            self.DL = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/idf.pkl', 'rb') as f:
            self.idf = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/cossim_denom_dict.pkl', 'rb') as f:
            self.w_doc = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/index_stem_body.pkl', 'rb') as f:
            self.index_stem_body = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/index_stem_title.pkl', 'rb') as f:
            self.index_stem_title = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/idf_stem.pkl', 'rb') as f:
            self.idf_stem = pickle.loads(f.read())
        with GCSFS.open('gs://315682872_209505593/cossim_denom_dict_stem.pkl', 'rb') as f:
            self.w_doc_stem = pickle.loads(f.read())
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tok = tokenize(query, False)
    body_scores = cossim_body(query_tok, app.index_body, app.DL, app.idf, app.w_doc, 100, 'postings_body')
    title_scores = search_title_helper(query_tok, app.index_title, 'postings_title')
    # dl = app.DL
    weight = 0.3
    merged_results = merge_scores(len(query_tok), title_scores, body_scores, app.page_rank, weight, (1 - weight), 20)
    res = [(x[0], app.titles.get(x[0], 'NO TITLE')) for x in merged_results]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tok = tokenize(query, False)
    cossim_result = cossim_body(query_tok, app.index_body, app.DL, app.idf, app.w_doc, 100, 'postings_body')
    res = [(x[0], app.titles.get(x[0], 'NO TITLE')) for x in cossim_result]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tok = tokenize(query, False)
    title_result = search_title_helper(query_tok, app.index_title, 'posting_titles')
    res = [(x[0], app.titles.get(x[0], 'NO TITLE')) for x in title_result]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tok = tokenize(query, False)
    title_result = search_anchor_helper(query_tok, app.index_anchor)
    res = [(x, app.titles.get(x, 'NO TITLE')) for x in title_result]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        res += [app.page_rank.get(doc_id, 0)]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        res += [app.page_view.get(doc_id, 0)]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
