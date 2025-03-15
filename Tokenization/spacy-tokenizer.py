import spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm

# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Making a list of the tokens and printing the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)

# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)