'''
FROM: https://github.com/gszsectan/SG-Trans/blob/
      2afab8844e4f1e06c06585d80158bda947e0c720/python/c2nl/tokenizers/tokenizer.py#L127
'''

class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()