import os
import pickle

class Cache:

    def __init__(self, cache_dir, query):

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.path = os.path.join(cache_dir, query + '.pkl')
        if os.path.exists(self.path):
            self.dict = pickle.load(open(self.path, 'rb'))
        else:
            self.dict = {}

    def dump_data(self, data):

        if os.path.exists(self.path):
            d = pickle.load(open(self.path, 'rb'))
        else:
            d = {}
        for k, v in data.items():
            d[k] = v
        pickle.dump(d, open(self.path, 'wb'))
    
    def load_data(self, k):
        
        d = pickle.load(open(self.path, 'rb'))
        return d[k]

    def verify_data(self, k):
        
        return k in self.dict
