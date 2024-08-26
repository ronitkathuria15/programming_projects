#!/usr/bin/env python
import sys
import string 

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

import gensim
import transformers 

from typing import List
from collections import defaultdict, OrderedDict

from sklearn.metrics.pairwise import cosine_similarity

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    result = set()
    words = wn.lemmas(lemma, pos=pos)
    for word in words:
        synsets = word.synset().lemmas()
        for synset in synsets:
            c = synset.name().replace('_', ' ')
            result.add(c)
    if lemma in result:
        result.remove(lemma)
    return list(result)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos
    synonym_frequencies = defaultdict(int)
    words = wn.lemmas(lemma, pos=pos)
    for word in words:
        synonyms = word.synset().lemmas()
        for synonym in synonyms:
             if synonym.name() != lemma:
                   formatted_synonym = synonym.name().replace('_', ' ')
                   synonym_frequencies[formatted_synonym] += synonym.count()
    most_frequent_synonym = max(synonym_frequencies, key=synonym_frequencies.get)
    return most_frequent_synonym

        

def wn_simple_lesk_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos
    stop_words = set(stopwords.words('english'))

    def normalize_word(words):
        return [word.lower() for word in words if word.lower() not in stop_words]
    
    def synset_tokens(synset):
        tokens = tokenize(synset.definition())
        for example in synset.examples():
            tokens += tokenize(example)
        return tokens

    def compute_aggregate_score(synset, target_lemma, candidate_lemma):
        overlap = len(set(normalized_tokens).intersection(set(normalized_context)))
        synset_freq_target = sum(lex_lem.count() for lex_lem in synset.lemmas() if lex_lem.name() == target_lemma)
        synset_freq_candidate = sum(lex_lem.count() for lex_lem in synset.lemmas() if lex_lem.name() == candidate_lemma)
        return 1000 * overlap + 100 * synset_freq_target + synset_freq_candidate
    
    result = {}
    for lexeme in wn.lemmas(lemma, pos=pos):
        synset = lexeme.synset()
        tokens = synset_tokens(synset)
        tokens += sum((synset_tokens(hypernym) for hypernym in synset.hypernyms()), [])
        
        word_context = context.left_context + context.right_context
        normalized_tokens = normalize_word(tokens)
        normalized_context = normalize_word(word_context)

        for candidate_lexeme in synset.lemmas():
            if candidate_lexeme.name() != lemma:
                formatted_synonym = candidate_lexeme.name().replace('_', ' ')
                score = compute_aggregate_score(synset, lemma, candidate_lexeme.name())
                result[formatted_synonym] = score

    if not result:
        return None

    sorted_candidates = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for candidate, _ in sorted_candidates:
        if candidate != lemma:
            return candidate

    return None  # If no suitable candidate is found, return None


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        possible_candidates = get_candidates(context.lemma, context.pos)
        similarity_scores = {}
        for synonym in possible_candidates:
            similarity_score = self.model.similarity(context.lemma, synonym) if synonym in self.model else 0
            similarity_scores[synonym] = similarity_score
        return max(similarity_scores, key=similarity_scores.get)

class BertPredictor(object):

    def __init__(self, context_window_size=5, confidence_threshold=0.2):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.context_window_size = context_window_size
        self.confidence_threshold = confidence_threshold

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        full_context = context.left_context + ["[MASK]"] + context.right_context
        input = self.tokenizer.encode(full_context)
        target_index = self.tokenizer.convert_ids_to_tokens(input).index("[MASK]")
        input_mat = np.array(input).reshape((1,-1))
        predictions = self.model.predict(input_mat, verbose=False)[0]
        best_words = np.argsort(predictions[0][target_index])[::-1]
        for word in self.tokenizer.convert_ids_to_tokens(best_words):
            if word in candidates:
                return word
        return None

    def special_predictor(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        # Adjust context to a fixed window size
        left_context = context.left_context[-self.context_window_size:]
        right_context = context.right_context[:self.context_window_size]
        full_context = left_context + ["[MASK]"] + right_context

        input = self.tokenizer.encode(full_context)
        target_index = self.tokenizer.convert_ids_to_tokens(input).index("[MASK]")
        input_mat = np.array(input).reshape((1, -1))
        predictions = self.model.predict(input_mat, verbose=False)[0]

        # Get top predictions and their probabilities
        top_indices = np.argsort(predictions[0][target_index])[::-1][:5]
        top_words = self.tokenizer.convert_ids_to_tokens(top_indices)
        top_probs = [predictions[0][target_index][i] for i in top_indices]

        for word, prob in zip(top_words, top_probs):
            if word in candidates and prob > self.confidence_threshold:
                return word
        return None



if __name__=="__main__":

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    bert = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = bert.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        #print(get_candidates('slow','a'))
