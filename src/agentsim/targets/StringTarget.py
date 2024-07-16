from agentsim.targets.BaseTarget import BaseTarget

class StringTarget(BaseTarget):
    
    def exact_str_match(self, pass_range, score):
        return pass_range == score
    
    # hamming distance
    
    # regex
    
    # levenshtein distance
    
    # jaccard similarity
    
    # cosine similarity
    
    def __init__(self, pass_range=None):
        in_range = self.exact_str_match
        super().__init__(pass_range, in_range)
