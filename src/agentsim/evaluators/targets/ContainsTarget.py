from agentsim.evaluators.targets.BaseTarget import BaseTarget

class ContainsTarget(BaseTarget):
    '''Evaluator for checking if an item is inside a set of passing items'''
    
    @staticmethod
    def contains(pass_range, x):
        '''Checks if x is in the pass_range'''

        return x in pass_range        
    
    def __init__(self, pass_range):
        super().__init__(pass_range, in_range = ContainsTarget.contains)