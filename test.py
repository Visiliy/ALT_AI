from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2_model")


text = "Thisisanexampleofmergedtext"

tokens = tokenizer.tokenize(text)
print(tokens)