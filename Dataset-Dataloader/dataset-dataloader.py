import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

ip_sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, ip_sentences))

# Create an instance of your custom data set
custom_dataset = CustomDataset(ip_sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
# print("Sample Items:")
# for i in range(6):
#     sample_item = custom_dataset[i]
#     print(f"Item {i + 1}: {sample_item}")

batch_size = 2
# Create a data loader with the custom collate function with batch_first=True,
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Iterate over the data loader
for i, batch in enumerate(dataloader):
    print(f"Batch {i + 1}:")
    print(batch)