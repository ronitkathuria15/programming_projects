import numpy as np
from hw5.utils import Node


class BayesNet:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.topological_sort()
        self.set_children()
    
    def topological_sort(self):
        new = []
        while self.nodes:
            for n in self.nodes:
                if set(n.parents) <= set(new):
                    new.append(n)
                    self.nodes.remove(n)
        self.nodes = new

    def set_children(self):
        for n in self.nodes:
            for p in n.parents:
                p.children.add(n)


    """
    4.1 Generate sample and weight from Bayes net by likelihood weighting
    """        
    def weighted_sample(self, evidence: dict={}):
        """
        Args:
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
            Float: Sample weight. 
        """
        sample = {}
        weight = 1
        #iterating through all the nodes in the bayes net
        for node in self.nodes:
            #used the observed value of the node if it's found in evidence
            if node in evidence:
                sample[node] = evidence[node]
                parent_values = [sample[parent] for parent in node.parents]
                #updating weight
                weight *= node.get_probs(parent_values)[sample[node]]
            else:
                parent_values = [sample[parent] for parent in node.parents]
                sample[node] = node.sample(parent_values)
        return sample, weight

    """
    4.2 Generate sample from Bayes net by Gibbs sampling
    """  
    def gibbs_sample(self, node: Node, sample: dict):
        """
        Args:
            node (Node): Node to resample.
            sample (Dict): {node:0/1} mappings for all nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
        """ 
        new = sample.copy()
        #computing Pr(X|MB(X))
        newParentValues = [new[parent] for parent in node.parents]
        prob0, prob1 = node.get_probs(newParentValues)
        #iterating through all the children of the node
        for child in node.children:
            parent_values = [new[parent] for parent in child.parents]
            #find the index of where the node is located
            indexNode = child.parents.index(node)
            #retaining the appropriate values by finding the children probabilities twice for X = 1 or 0
            prob0 *= child.get_probs(parent_values)[new[child]]
            parent_values[indexNode] = 0
            prob1 *= child.get_probs(parent_values)[new[child]]
        #normalizing
        prob = (prob0, prob1)
        prob = np.array(prob) / np.sum(prob)
        #resampling X
        new[node] = np.random.choice([0,1], p=prob)
        return new

    """
    4.3 Generate a list of samples given evidence and estimate the distribution.
    """  
    def gen_samples(self, numSamples: int, evidence: dict={}, LW: bool=True):
        """
        Args:
            numSamples (int).
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
            LW (bool): Use likelihood weighting if True, Gibbs sampling if False.
        Returns:
            List[Dict]: List of {node:0/1} mappings for all nodes.
            List[float]: List of corresponding weights.
        """       
        samples = []
        weights = []
            #using likelihood weighting
        if LW:
            #iterate through the number of samples
            for i in range(numSamples):
                sample, weight = self.weighted_sample(evidence)
                #adding samples to list
                samples.append(sample)
                weights.append(weight)
            #using gibbs 
        else:
            #using weighted_sample to generate the initial sample 
            sample, _ = self.weighted_sample(evidence) 
            #declaring counter to compare against numSamples  
            sampleCount = 0
            while sampleCount < numSamples:
                for node in self.nodes:
                    if node not in evidence:
                        sample = self.gibbs_sample(node, sample)
                        sampleCount += 1
                        samples.append(sample)
            weights = [1] * numSamples
        return samples, weights
                

    def estimate_dist(self, node: Node, samples: list[dict], weights: list[float]):
        """
        Args:
            node (Node): Node whose distribution we will estimate.
            samples (List[Dict]): List of {node:0/1} mappings for all nodes.
            weights (List[float]: List of corresponding weights.
        Returns:
            Tuple(float, float): Estimated distribution of node variable.
        """           
        prob0, prob1 = 0,0
        #iterating through each sample 
        for i in range(len(samples)):
            if samples[i][node] == 0:
                prob0 += weights[i]
            elif samples[i][node] == 1:
                prob1 += weights[i]
        total_weights = prob0 + prob1
        prob0 /= total_weights
        prob1 /= total_weights
        return (prob0,prob1)