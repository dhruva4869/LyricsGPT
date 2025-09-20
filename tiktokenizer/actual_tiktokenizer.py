import tiktoken
text = "Hello, world! @&$fh1647284638         def __init__(self, name, age): somethi g g g <|endoftext|>"
enc = tiktoken.get_encoding("cl100k_base")
res = enc.encode(text, allowed_special={"<|endoftext|>"})
for num in res:
    print(f" == {num} --> {enc.decode([num])} == ")

print(enc.decode(enc.encode(text, allowed_special={"<|endoftext|>"})))