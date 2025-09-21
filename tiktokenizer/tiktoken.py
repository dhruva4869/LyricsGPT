import regex as re
from collections import Counter

class TikToken:
    def __init__(self, text, vocab_size):
        self.text = text
        self.vocab_size = vocab_size
        self.pattern_match = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = re.compile(self.pattern_match)
        self.ranks = {} # chr to ord
        self._decoder = {} # ord to chr
        self.next_token = 256
        self._init_byte_tokens()
    
    def _init_byte_tokens(self, debug=False):
        for i in range(self.next_token):
            token_bytes = bytes([i])
            self.ranks[token_bytes] = i
            self._decoder[i] = token_bytes
            if debug:
                # this is basically token_bytes = chr and i = ord
                print(f"Token bytes : {token_bytes}==\nLatest token used : {i}")
    
    def train(self):
        words = self.pattern.findall(self.text)
        # print(words)
        word_byte_sequences = []
        for word in words:
            word_byte = word.encode('utf-8')
            word_byte_sequences.append([bytes([b]) for b in word_byte])
        # print(word_byte_sequences)

        while len(self.ranks) < self.vocab_size:
            pair_counts = Counter()
            for word_bytes in word_byte_sequences:
                for i in range(len(word_bytes)-1):
                    pair_counts[(word_bytes[i-1], word_bytes[i])] += 1

            if not pair_counts:
                break

            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            new_token_bytes = most_frequent_pair[0] + most_frequent_pair[1]
            self.ranks[new_token_bytes] = self.next_token
            self._decoder[self.next_token] = new_token_bytes
            self.next_token += 1

            new_word_byte_sequences = []
            for word_bytes in word_byte_sequences:
                new_word = []
                i = 0
                while i < len(word_bytes) - 1:
                    if (word_bytes[i-1], word_bytes[i]) == most_frequent_pair:
                        new_word.append(new_token_bytes)
                        i += 2
                    else:
                        new_word.append(word_bytes[i])
                        i += 1
                if i == len(word_bytes) - 1:
                    new_word.append(word_bytes[i])
                new_word_byte_sequences.append(new_word)
            
            word_byte_sequences = new_word_byte_sequences
        
        return self.ranks
    
    def encode(self, text):
        words = self.pattern.findall(text)
        tokens = []
        for word in words:
            word_bytes = word.encode("utf-8")
            word_tokens = self._bpe_encode_word(word_bytes)
            tokens.extend(word_tokens)
        
        return tokens
    
    def _bpe_encode_word(self, word_bytes):
        parts = [bytes([b]) for b in word_bytes]
        while True:
            best_pair = None
            best_rank = None
            
            for i in range(len(parts) - 1):
                pair_bytes = parts[i] + parts[i + 1]
                if pair_bytes in self.ranks:
                    rank = self.ranks[pair_bytes]
                    if best_rank is None or rank < best_rank:
                        best_pair = i
                        best_rank = rank
            
            if best_pair is None:
                break
                
            merged = parts[best_pair] + parts[best_pair + 1]
            parts = parts[:best_pair] + [merged] + parts[best_pair + 2:]

        return [self.ranks[part] for part in parts]

    def decode(self, tokens):
        byte_sequence = b"".join(self._decoder[token] for token in tokens)
        return byte_sequence.decode("utf-8", errors="replace")

    def decode_(self, tokens):
        return [self._decoder[token] for token in tokens]

    def visualize_encoding(self, text):
        tokens = self.encode(text)
        token_bytes = self.decode_(tokens)
        
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token bytes: {[tb.decode('utf-8', errors='replace') for tb in token_bytes]}")
        print(f"Decoded: '{self.decode(tokens)}'")
        print("-" * 50)


def test_tokenizer():
    """Test the improved tokenizer with various inputs."""

    test_texts = [
        "Hello world! 🌍",
        "I love emojis! 😀😃😄😁😆😅😂🤣",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "Numbers: 123 456.789",
        "Unicode: café naïve résumé",
        "Mixed: Hello123! @#$% 😀",
        "Spaces and\ttabs\nand newlines",
        "Apostrophes: don't can't won't",
        "Quotes: \"double\" and 'single'",
    ]
    
    tokenizer = TikToken(" ".join(test_texts), vocab_size=500)
    tokenizer.train()
    
    print("=== TikTokenizer Test ===")
    print(f"Vocabulary size: {len(tokenizer.ranks)}")
    print()
    
    for text in test_texts:
        tokenizer.visualize_encoding(text)
    
    print("\n=== Round-trip Test ===")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Decoded:  '{decoded}'")
        print(f"Match: {text == decoded}")
        assert text == decoded, "Text did not match"
        print()


if __name__ == "__main__":
    test_tokenizer()