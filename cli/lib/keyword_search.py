from lib.search_utils import (
    load_movies,
    load_stopwords,
    CACHE_PATH
    )
import re 
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os 
import pickle
import math

#params
BM25_k1 = 1.5
BM25_B = 0.75

def clean(string):
    string = string.lower() #case sensitivity 
    # Keep all whitespace (including newlines) so removing punctuation never
    # joins two otherwise separate tokens, e.g. "love.\nWhen".
    string = re.sub(r"[^a-zA-Z0-9\s]", "", string) #punctuation removal
    return string

def tokenize(string,stopwords,stemmer):
    string = clean(string)
    #stopwords removal also
    tokens = [tok for tok in string.split() if tok and tok not in stopwords]
    #stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def has_matching_tokens(query,title):
    for query_tok in query:
        for title_tok in title:
            if query_tok in title_tok:
                return True
    return False

class InvertedIndex: 
    def __init__(self,):
        self.index = defaultdict(set)
        self.docmap = {}
        self.doclens = {}
        self.index_path = CACHE_PATH/"index.pkl"
        self.docmap_path = CACHE_PATH/"docmap.pkl"
        self.doclens_path = CACHE_PATH/"doclens.pkl"
        self.term_frequencies_path = CACHE_PATH/"term_frequencies.pkl"
        self.stopwords = load_stopwords()
        self.stemmer = PorterStemmer()
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self,doc_id,text):
        tokens = tokenize(text, stopwords= self.stopwords,stemmer=self.stemmer)
        #cnt = Counter()
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

        length = len(tokens)
        self.doclens[doc_id] = length

    def __get_avg_doc_length(self):
        num_docs = len(self.doclens)
        if num_docs:
            avg_doc_length = sum(self.doclens.values())/num_docs
            return avg_doc_length
        else:
            return 0.0

    def get_documents(self, term):
        ids = sorted(list(self.index[term]))
        return ids

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(doc_id,text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies,f)

        with open(self.doclens_path, "wb") as f:
            pickle.dump(self.doclens, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doclens_path, "rb") as f:
                self.doclens = pickle.load(f)
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Problem loading pickle: {e}")

    def get_tf(self,doc_id,term): 
        term = tokenize(term,stopwords= self.stopwords,stemmer=self.stemmer)
        if len(term)!= 1:
            raise ValueError("Expected only one term got more :(")
        if term[0] in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][term[0]]
        else:
            return 0

    def get_idf(self,term):
        query = tokenize(term, stopwords=self.stopwords,stemmer=self.stemmer)
        n = len(self.docmap)
        n_term = len(self.index[query[0]])
        idf = math.log((n+1)/(n_term+1))
        return idf

    def get_tfidf(self,doc_id,term):
        term = tokenize(term, stopwords=self.stopwords,stemmer=self.stemmer)
        tf = self.get_tf(doc_id,term[0])
        idf = self.get_idf(term[0])
        return tf*idf

    def get_bm25_tf(self, doc_id, term, k1 = BM25_k1,b=BM25_B):
        tf = self.get_tf(doc_id=doc_id,term=term)
        doc_len = self.doclens[doc_id]
        avg_doc_len = self.__get_avg_doc_length()
        len_norm = 1-b + b*(doc_len/avg_doc_len)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1*len_norm)
        return bm25_tf

    def get_bm25_idf(self, term: str) -> float:
        query = tokenize(term, stopwords=self.stopwords,stemmer=self.stemmer)
        n = len(self.docmap) #total docs
        df = len(self.index[query[0]])#hit docs
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return idf

    def bm25(self,doc_id,term,k1=BM25_k1,b=BM25_B):
        tf = self.get_bm25_tf(doc_id,term,k1,b)
        idf = self.get_bm25_idf(term)
        bm25 = tf*idf
        return bm25

    def bm25search(self,query,limit=5, k1=BM25_k1,b=BM25_B):
        query = tokenize(query, stopwords=self.stopwords,stemmer=self.stemmer)
        bm25_scores = {}
        #bm25score for a token -> repeat for entire query -> sum for score for one doc
        #repeat this for all docs
        for token in query:
            doc_ids = self.index[token]
            for doc_id in doc_ids:
                bm25_token_score = self.bm25(doc_id=doc_id,term=token)
                bm25_scores[doc_id] = bm25_scores.get(doc_id,0) + bm25_token_score

        #sort the dictionary and return top limit results
        bm25_scores = dict(sorted(bm25_scores.items(), key=lambda x: x[1], reverse= True))
        results = []
        counter = 1
        for id,score in bm25_scores.items():
            title = self.docmap[id]["title"]
            results.append(f"{id} {title} - Score: {score:.2f}")
            if counter == limit:
                break
            counter+=1
        return results  
            
def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    id = idx.get_tf(doc_id,term)
    return id

def search_command(query, n_results):
    movies = load_movies()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    idx = InvertedIndex()
    idx.load()
    seen,res = set(),[]
    query = tokenize(query,stopwords,stemmer)

    for token in query:
        matching_doc_ids = idx.get_documents(token)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            else:
                matching_doc = idx.docmap[matching_doc_id]
                res.append(matching_doc["title"])
                seen.add(matching_doc_id)
            if len(res)>= n_results:
                return res
    return res

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    return idf
   
def tfidf_command(doc_id,term):
    idx = InvertedIndex()
    idx.load()
    tfidf = idx.get_tfidf(doc_id=doc_id, term=term)
    return tfidf

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_bm25_idf(term=term)
    return idf

def bm25_tf_command(doc_id,term, k1 = BM25_k1):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id=doc_id,term=term,k1=k1)
    return bm25_tf

def bm25_search(query,limit=5):
    idx = InvertedIndex()
    idx.load()
    bm25scores = idx.bm25search(query=query,limit=limit)
    return bm25scores