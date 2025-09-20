from collections import defaultdict


TEXT = "Hello everyone Here we have space e and then a space again many times 11"
TEXT = """
You're now ready to load the merges from the GPT-4 tokenizer and show that your tokenizer produces the identical results for both encode and decode, matching tiktoken.

# match this
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("hello world!!!? lol123 ")
text = enc.decode(ids) # get the same text back

def __init(self):
    pass


"""

class SimpleBitEncode:
    def __init__(self, text: str):
        self.text = text

    def encode(self):
        return list(map(int, self.text.encode(errors='replace')))

    def decode(self):
        return "".join(list(map(chr, self.encode(errors='replace'))))
    
SimpleEncoder = SimpleBitEncode(TEXT)
print(SimpleEncoder.encode())
# print(SimpleEncoder.decode())



class NaiveBPE(SimpleBitEncode):
    def __init__(self, text: str):
        super().__init__(text)
        self.text = text
        self.most_common_pairs = defaultdict(int)
        self.next_token = None
        self.pair_counts = None
        self.encoded = None
        self.chars_to_token = {}
        self.fetch = {}
        self.pair_to_new_token = {}
    
    def preprocess(self, pair_counts=2):
        encoded = self.encode()
        self.encoded = encoded
        self.next_token = self.next_token or max(encoded) + 1
        self.pair_counts = pair_counts
        for i in range(len(encoded)-1):
            self.most_common_pairs[(encoded[i], encoded[i+1])] += 1 
        
        self.most_common_pairs = dict(sorted(self.most_common_pairs.items(), key = lambda x : -x[1]))
        return self.most_common_pairs

    def encode_bpe(self):
        self.preprocess()
        pair_counts = self.pair_counts
        pair_to_new_token = defaultdict(int)
        self.pair_to_new_token = pair_to_new_token
        for k, v in self.most_common_pairs.items():
            pair_to_new_token[k] = self.next_token
            self.next_token += 1
            pair_counts -= 1
            if pair_counts == 0:
                break
        
        # print(pair_to_new_token)
        chars_to_new_token = {chr(k[0]) + chr(k[1]) : v for k, v in pair_to_new_token.items()}
        # print(chars_to_new_token)
        self.chars_to_token = chars_to_new_token

        new_bpe = []
        for i in range(len(self.encoded)-1):
            if (self.encoded[i], self.encoded[i+1]) in pair_to_new_token:
                new_bpe.append(pair_to_new_token[(self.encoded[i], self.encoded[i+1])])
            else:
                new_bpe.append(self.encoded[i])
        new_bpe.append(self.encoded[-1])
        return new_bpe
    
    def decode_bpe(self, bpe):
        self.fetch = {v : k for k, v in self.chars_to_token.items()}
        print(self.fetch)
        res = []
        i = 0
        n = len(bpe)
        while i < n:
            encoded = bpe[i]
            if encoded < min(self.pair_to_new_token.values()):
                res.append(chr(encoded))
                i += 1
            else:
                res.append(self.fetch[encoded])
                i += 2
        return "".join(res)

        


BPE = NaiveBPE(TEXT)
BPE.preprocess()
print(BPE.most_common_pairs)
res = BPE.encode_bpe()
print(res)
res = BPE.decode_bpe(res)
print(res)
assert res == TEXT, "Decoding went wrong somewhere"