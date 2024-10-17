import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import nltk
from nltk.corpus       import stopwords

import re

from gensim            import  corpora
from gensim.models     import  LsiModel
from gensim.models     import  LdaModel
from gensim.models     import  CoherenceModel
from gensim.models     import  TfidfModel

import warnings
warnings.filterwarnings("ignore")



class Texto:
    
    def __init__(self, textos:list, palabras_parada:list):
        self.textos = textos
        self.palabras_parada = palabras_parada
        self.textos_limpios = self.limpieza(textos)
        self.union_textos = " ".join(self.textos_limpios)
        self.diccionario = self.dictionary_corpus(self.textos_limpios)
        self.corpus = self.corpus_count(self.textos_limpios)
        self.corpus_tfidf = self.gen_corpus_tfidf(self.textos_limpios)

    def limpieza(self, textos):
        textos_limpios = []
        for texto in textos:
            texto_min = texto.lower()
            texto_sin_esp = re.sub(r'[^\w\s]', '', texto_min)
            texto_sin_num = re.sub(r'\b[0-9]+\b', '', texto_sin_esp)
            texto_sin_tildes = re.sub("[áéíóú]", lambda x: {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}[x.group()], texto_sin_num)
            palabras = texto_sin_tildes.split()

            stop_words = stopwords.words('spanish')
            stop_words.extend(self.palabras_parada)
            stop_words_sin_tilde = [re.sub("[áéíóú]", lambda x: {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}[x.group()], word) for word in stop_words if word != 'no']
            words_cadena = [word for word in palabras if (word not in stop_words_sin_tilde and len(word) > 3) or word == 'no']
            
            cadena_final = ' '.join(words_cadena)
            textos_limpios.append(cadena_final)
        return textos_limpios
    
    def palabras_mas_comunes(self, n_palabras:int, plot:bool):
        palabras = self.union_textos.split()
        frecuencia = nltk.FreqDist(palabras)
        palabras_mas_comunes = frecuencia.most_common(n_palabras)

        if plot:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=[palabra[0] for palabra in palabras_mas_comunes], y=[palabra[1] for palabra in palabras_mas_comunes], palette='Spectral')
            plt.title('15 palabras más comunes en titulares de El Tiempo')
            plt.xticks(rotation=45)
            plt.show()

        return palabras_mas_comunes
    
    def doc_tok(self, textos):
        return [texto.split() for texto in textos]

    def dictionary_corpus(self, textos):
        doc_tok = [texto.split() for texto in textos]
        dictionary = corpora.Dictionary(doc_tok)
        return dictionary

    def corpus_count(self, textos):
        dictionary = self.dictionary_corpus(textos)
        doc_tok = [texto.split() for texto in textos]
        corpus = [dictionary.doc2bow(doc) for doc in doc_tok]
        return corpus
    
    def gen_corpus_tfidf(self, textos):
        corpus = self.corpus_count(textos)
        tfidf = TfidfModel(corpus)
        return tfidf[corpus]
    


def coherence_model(corpus, dictionary, doc_tok, lsi:bool, n_topics:list):
    if lsi:
        cms = [CoherenceModel(model= LsiModel(corpus=corpus,
                                       num_topics=i,
                                       id2word=dictionary),
                       texts     = doc_tok,
                       corpus    = corpus,
                       coherence = 'c_v') for i in n_topics] 

        coherences = [cm.get_coherence() for cm in cms]

        r = sorted(dict(zip(n_topics, coherences)).items(), key=lambda item: item[1], reverse=True)
    else:
        cms =  [CoherenceModel(model= LdaModel(corpus=corpus,
                                       num_topics=i,
                                       id2word=dictionary),
                       texts     = doc_tok,
                       corpus    = corpus,
                       coherence = 'c_v') for i in n_topics]

        coherences = [cm.get_coherence() for cm in cms]
        r = sorted(dict(zip(n_topics, coherences)).items(), key=lambda item: item[1], reverse=True)
    return r


def finding_model(corpus, dictionary, doc_tok, lsi:bool, n_top:int, min_topics:int):
    n_topics = [i for i in range(1,n_top+1)]
    if lsi:
        r = coherence_model(corpus, dictionary, doc_tok, lsi, n_topics)
    else:
        r = coherence_model(corpus, dictionary, doc_tok, lsi, n_topics)

    for i in r:
        if i[0] >= min_topics:
            return i
    
