from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    
        while state.buffer: 
            # TODO: Write the body of this loop for part 4 
            valid_shift = True
            valid_left = True
            valid_right = True
            transitions = self.model.predict((self.extractor.get_input_representation(words, pos, state)).reshape(1,-1))[0]
            if len(state.stack) == 0:
                valid_left = False
                valid_right = False
            else:
                if len(state.buffer) == 1:
                    valid_shift = False
                if state.stack[-1] == 0:
                    valid_left = False
            while True:
                max_index = np.argmax(transitions)
                chosen_transition, chosen_tag = self.output_labels[max_index]
                transitions[max_index] = -np.inf  # Invalidate this prediction

                if chosen_transition == 'shift' and valid_shift:
                    state.shift()
                    break
                elif chosen_transition == 'left_arc' and valid_left:
                    state.left_arc(chosen_tag)
                    break
                elif chosen_transition == 'right_arc' and valid_right:
                    state.right_arc(chosen_tag)
                    break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
