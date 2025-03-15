# nltk.download("punkt")
# nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize


# Use nltk
text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
tokens = word_tokenize(text)
print(tokens)