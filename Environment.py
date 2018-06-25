

# I should have written a metaclass to make sure all the subclass follow the rule.

# This is an abstract class for environment. The invironments in all the cases would follow this class. 
class Environment():
    def __init__(self):
        pass
    def Reset(self):
        pass
    def Step(self,action):
        raise NotImplementedError()
    
    

         