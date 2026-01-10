import os
import spacy
import torch

from torchvision.io import read_image
from torch.utils.data import Dataset


spacy_end = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_end.tokenizer(text)]

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 # continue from the 4 special tokens

        # Store words and their frequencies
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # Adding word to the vocabulary if its frequencies == freq_threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1


class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_df, vocabulary, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.captions = captions_df
        self.vocabulary = vocabulary
        self.transform = transform
        self.target_transform = target_transform
        self.col_idx = self.captions.columns.get_indexer(["image_name", " comment"])

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.captions.iloc[idx, self.col_idx[0]])
        image = read_image(image_path)
        image_id = self.captions.iloc[idx, self.col_idx[0]]
        raw_caption = self.captions.iloc[idx, self.col_idx[1]]

        if self.transform:
            image = self.transform(image)

        # Adding <SOS> and <EOS> to the caption
        caption = [self.vocabulary.stoi["<SOS>"]]
        caption += self.vocabulary.numericalize(raw_caption)
        caption.append(self.vocabulary.stoi["<EOS>"])

        return image, torch.tensor(caption), image_id
    