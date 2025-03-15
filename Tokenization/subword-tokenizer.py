from transformers import BertTokenizer
from transformers import XLNetTokenizer



# Wordpiece
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.tokenize("IBM taught me tokenization."))

# Unigram and SentencePiece
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
print(tokenizer.tokenize("IBM taught me tokenization."))