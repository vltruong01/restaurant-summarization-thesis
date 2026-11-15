"""Exact CPU token-count memoization for length-only diagnostics, never models."""
class LengthOnlyIds:
    def __init__(self,count):self.count=count
    def __len__(self):return self.count
    def __iter__(self):raise TypeError('Length-only diagnostics do not expose token IDs')
    def __getitem__(self,index):raise TypeError('Length-only diagnostics do not expose token IDs')


class LengthOnlyEncoding:
    def __init__(self,count):self.ids=LengthOnlyIds(count)


class TokenLengthCache:
    """Cache exact encoder counts by full unchanged string and special-token flag.

    Scientific scripts currently consume only len(encoding.ids). Accessing token
    IDs intentionally fails. Use the actual Tokenizer for training/input IDs.
    """
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.cache={};self.hits=0;self.misses=0
    def encode(self,text,add_special_tokens=True):
        key=(text,add_special_tokens)
        if key not in self.cache:
            self.cache[key]=len(self.tokenizer.encode(text,add_special_tokens=add_special_tokens).ids)
            self.misses+=1
        else:self.hits+=1
        return LengthOnlyEncoding(self.cache[key])
