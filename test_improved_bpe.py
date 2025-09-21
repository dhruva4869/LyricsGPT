#!/usr/bin/env python3
"""
Test script for the improved BPE tokenizer.
"""

from tiktokenizer.tiktoken_gpt2 import ImprovedBPE

def main():
    print("=== Testing Improved BPE Tokenizer ===\n")
    
    # Test with a simple example first
    simple_text = "Hello world! 🌍 This is a test with emojis 😀 and special chars @#$%"
    
    # Create and train tokenizer
    tokenizer = ImprovedBPE(simple_text, vocab_size=300)
    tokenizer.train()
    
    print(f"Vocabulary size: {len(tokenizer.mergeable_ranks)}")
    print()
    
    # Test encoding and decoding
    print("=== Simple Test ===")
    tokens = tokenizer.encode(simple_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: '{simple_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    print(f"Perfect match: {simple_text == decoded}")
    print()
    
    # Test with more complex examples
    test_cases = [
        "Emojis: 😀😃😄😁😆😅😂🤣",
        "Unicode: café naïve résumé",
        "Mixed: Hello123! @#$% 😀",
        "Apostrophes: don't can't won't",
        "Quotes: \"double\" and 'single'",
        "Numbers: 123 456.789",
        "Special: @#$%^&*()_+-=[]{}|;':\",./<>?",
    ]
    
    print("=== Complex Test Cases ===")
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        match = test_text == decoded
        
        print(f"  Original: '{test_text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {match}")
        
        if not match:
            print(f"  ⚠️  Round-trip failed!")
        else:
            print(f"  ✅ Perfect round-trip!")

if __name__ == "__main__":
    main()
