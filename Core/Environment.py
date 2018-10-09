from abc import ABC, abstractmethod

# I should have written a metaclass to make sure all the subclass follow the rule.
# This meta class forces the subclass to have the "features_size" attribute, "action_space" attribute.


# This is an abstract class for environment. The environments in all the cases would follow this class.
class Environment(ABC):
    def __init__(self, actions_num, features_dim):
        self.actions_num = actions_num
        self.features_dim = features_dim

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()
