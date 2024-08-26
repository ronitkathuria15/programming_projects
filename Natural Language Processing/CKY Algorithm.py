import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        parse_table = defaultdict(set)
        num = len(tokens)

        for i in range(num):
            for rule in self.grammar.rhs_to_rules.get((tokens[i],), []):
                parse_table[(i, i+1)].add(rule[0])
        
        for length in range(2, num+1):
            for i in range(num-length+1):
                j = i + length
                for k in range(i+1, j):
                    for a in parse_table[(i, k)]:
                        for b in parse_table[(k, j)]:
                            rules = {rule[0] for rule in self.grammar.rhs_to_rules.get((a, b), [])}
                            parse_table[(i, j)].update(rules)  
                                     
        return self.grammar.startsymbol in parse_table[(0,num)]  
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        num = len(tokens)
        table= defaultdict(dict)
        probs = defaultdict(dict)

        for i in range(num):
            for rule in self.grammar.rhs_to_rules.get((tokens[i],), []):
                probability = rule[2]
                lhs = rule[0]
                table[(i, i+1)][lhs] = tokens[i]
                probs[(i, i+1)][lhs] = math.log(probability)
        
        for length in range(2, num+1):
            for i in range(num-length+1):
                j = i + length
                for k in range(i+1, j):
                    for a in table[(i, k)]:
                        for b in table[(k, j)]:
                            for rule in self.grammar.rhs_to_rules.get((a, b), []):
                                lhs, rhs, probability = rule
                                combined_probability = math.log(probability) + probs[(i, k)][rhs[0]] + probs[(k, j)][rhs[1]]
                                if (lhs not in probs[(i,j)] or combined_probability > probs[(i,j)][lhs]):
                                        probs[(i,j)][lhs] = combined_probability
                                        table[(i,j)][lhs] = ((a,i,k), (b,k,j))

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if type(chart[(i, j)][nt]) is str:
        return (nt, chart[(i, j)][nt])
    left_subtree = chart[(i, j)][nt][0]
    right_subtree = chart[(i, j)][nt][1]
    tree1 = get_tree(chart, left_subtree[1], left_subtree[2], left_subtree[0]) 
    tree2 = get_tree(chart, right_subtree[1], right_subtree[2], right_subtree[0])
    return (nt, tree1 ,tree2 )
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        #toks1 =['miami', 'flights','cleveland', 'from', 'to','.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        #print(check_table_format(table))
        #print(check_probs_format(probs))
        #print(get_tree(table, 0, len(toks), grammar.startsymbol))
        assert check_table_format(table)
        assert check_probs_format(probs)
        
