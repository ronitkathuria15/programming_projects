import sys
from collections import defaultdict
import math
import random
import os
import os.path

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    result = []

    #Adding START and STOP markers to the given sequence
    if n > 1:
        marked_sequence = (['START'] * (n - 1)) + sequence + ['STOP']
    else:
        marked_sequence = ['START'] + sequence + ['STOP']

    num_ngrams = len(marked_sequence) - n + 1

    #Generating the Python tuples
    for x in range(num_ngrams):
        ngram = tuple(marked_sequence[x:x + n])
        result.append(ngram)

    return result

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.total_sentencecount = 0
        self.count_ngrams(generator)
        self.total_wordcount = sum(self.unigramcounts.values())

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 

        #Generating ngrams 
        for sentence in corpus:
            self.total_sentencecount += 1
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            #Computing their respective counts
            for i in unigrams:
                if (i[0] != ('START',)):
                    self.unigramcounts[i] += 1

            for i in bigrams:
                if (i[0] != ('START','START')):
                    self.bigramcounts[i] += 1

            for i in trigrams:
                self.trigramcounts[i] += 1

        return


    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        num_trigram = self.trigramcounts[trigram]

        #Using the trigram to find the denominator
        previous_bigram = (trigram[0], trigram[1])
        num_bigram = self.bigramcounts[previous_bigram]

        #Checking if the context has been observed
        if previous_bigram == ('START','START'):
            num_bigram = self.total_sentencecount
        elif num_bigram == 0:
           return 1/(len(self.lexicon) - 1)
        result = num_trigram/num_bigram
        return result

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        num_bigram = self.bigramcounts[bigram]

        #Using the bigram to find the denominator
        previous_unigram = (bigram[0],)
        if previous_unigram == ('START',):
            num_unigram = self.total_sentencecount
        else:
            num_unigram = self.unigramcounts[previous_unigram]
        result = num_bigram/num_unigram
        return result
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #Computing the raw unigram probability using the counter in the model instance
        if unigram == ('START',):
            result = 0.0
        else:
            result = self.unigramcounts[unigram] / self.total_wordcount 
        return result

#     def generate_sentence(self,t=20): 
#         """
#         COMPLETE THIS METHOD (OPTIONAL)
#         Generate a random sentence from the trigram model. t specifies the
#         max length, but the sentence may be shorter if STOP is reached.
#         """
#         return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        #Computing smoothed probabilities
        trigram_probability_estimate = lambda1 * self.raw_trigram_probability(trigram)
        bigram_probability_estimate = lambda2 * self.raw_bigram_probability(trigram[1:])
        unigram_probability_estimate = lambda3 * self.raw_unigram_probability((trigram[2],))

        result = trigram_probability_estimate + bigram_probability_estimate + unigram_probability_estimate
        return result
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        logprobability = 0.0

        #Computing trigrams
        trigrams = get_ngrams(sentence, 3)

        #Computing the log probabilities for each trigram in the sequence and then summing
        for trigram in trigrams:
            if self.smoothed_trigram_probability(trigram) > 0:
                trigram_logprobability = math.log2(self.smoothed_trigram_probability(trigram))
            logprobability += trigram_logprobability
        
        return logprobability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0.0
        M = 0

        #Summing the log probabilities for each sentence and then dividing by the number of tokens
        for sentence in corpus:
            M += len(sentence) + 1
            sentence_probaility = self.sentence_logprob(sentence)
            l += sentence_probaility
        average_l = l/M
        result = 2**(-average_l)
        return result


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        #Computing and comparing the perplexities for each directory and model
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp2 < pp1:
                correct += 1
            total += 1

        accuracy = correct/total

        return accuracy

#if __name__ == "__main__":

#    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
# Testing perplexity: 
#dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
#pp = model.perplexity(dev_corpus)
#print(pp)


#Essay scoring experiment: 
#acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
#print(acc)

# sequence = ["natural", "language", "processing"]
# n = 1
# result = get_ngrams(sequence, n)
# print(result)

# if __name__ == "__main__":
#     training_corpus_file = "hw1_data/brown_train.txt"
#     model = TrigramModel(training_corpus_file)

#     training_corpus = corpus_reader(training_corpus_file, model.lexicon)
#     training_perplexity = model.perplexity(training_corpus)
#     print("Training Perplexity:", training_perplexity)

#     test_corpus_file = "hw1_data/brown_test.txt"
#     test_corpus = corpus_reader(test_corpus_file, model.lexicon)
#     test_perplexity = model.perplexity(test_corpus)
#     print("Test Perplexity:", test_perplexity)

model = TrigramModel('hw1_data/brown_train.txt')

#print(model.trigramcounts[('START', 'START', 'the')])
#print(model.bigramcounts[('START','the')])
#print(model.unigramcounts[('the',)])

#trigram = ('the', 'boy', 'sleeps')  
#smooth_trigram_prob = model.smoothed_trigram_probability(trigram)
#print(f'Smooth Trigram Probability: {smooth_trigram_prob}')

test_sentence = ["This", "is", "a", "test", "sentence"]
log_prob = model.sentence_logprob(test_sentence)
print("Log Probability:", log_prob)

# print(model.raw_trigram_probability(('START','START','the'))) # Part 3
# print(model.raw_bigram_probability(('START','the'))) # Part 3
# print(model.raw_unigram_probability(('START'))) # Part 3
# print(model.smoothed_trigram_probability(('START','START','the'))) # Part 4
# print(model.sentence_logprob(["natural","language","processing"])) # Part 5


