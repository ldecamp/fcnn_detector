""" Define base class for Building Deep Models
"""

class ModelBuilderBase(object):

    def __init__(self, ns=""):
        self.ns = ns
    
    def get_model(self):
        raise NotImplementedError()
    
    def get_output_shape(self):
        raise NotImplementedError()
