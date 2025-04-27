import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter, defaultdict
import spacy
import random
from utils.batch import Batch

# 请确保在运行前安装好 spaCy 模型：
# !python -m spacy download de_core_news_sm
# !python -m spacy download en_core_web_sm

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# Tokenization

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Special tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNK_WORD = '<unk>'
MIN_FREQ = 2
MAX_LEN = 100

# Vocab class
class Vocab:
    def __init__(self, counter, min_freq=1, specials=None):
        specials = specials or []
        self.itos = list(specials)
        token_set = set(specials)
        self.counter = counter 
        for token, freq in counter.items():
            if freq >= min_freq and token not in token_set:
                self.itos.append(token)
        # 自动将不在词表中的 token 映射为 <unk>
        self.stoi = defaultdict(lambda: self.stoi[UNK_WORD])
        for i, token in enumerate(self.itos):
            self.stoi[token] = i

    def __getitem__(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.itos)

    def freq(self, token):
        return self.counter[token]

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Preprocessing

def preprocess(example):
    src_tokens = tokenize_de(example["de"])
    tgt_tokens = tokenize_en(example["en"])
    if len(src_tokens) <= MAX_LEN and len(tgt_tokens) <= MAX_LEN:
        return {
            "src": src_tokens,
            "trg": [BOS_WORD] + tgt_tokens + [EOS_WORD]
        }
    return None

def filter_none(example):
    return example is not None

# Build vocab

def build_vocab(data, key):
    counter = Counter()
    for ex in data:
        counter.update(ex[key])
    return Vocab(counter, min_freq=MIN_FREQ, specials=[BLANK_WORD, UNK_WORD, BOS_WORD, EOS_WORD])

# Encode

def encode(example, src_vocab, tgt_vocab):
    example["src_ids"] = [src_vocab[token] for token in example["src"]]
    example["trg_ids"] = [tgt_vocab[token] for token in example["trg"]]
    return example

# Batch Sampler

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, pool_size=100, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))#每个句子的索引列表
        if self.shuffle:
            random.shuffle(indices)#打乱索引列表

        pooled_batches = []
        for i in range(0, len(indices), self.batch_size * self.pool_size):
            pool = indices[i: i + self.batch_size * self.pool_size] #大池中的样本索引
            pool = sorted(pool, key=lambda x: self.lengths[x])#x是pool中的索引，self.lengths[x]是句子的长度

            batches = [pool[j: j + self.batch_size] for j in range(0, len(pool), self.batch_size)] #将池中的样本分成小批次
            if self.shuffle:
                random.shuffle(batches)#小池中打乱索引顺序
            pooled_batches.extend(batches)

        return iter([idx for batch in pooled_batches for idx in batch])#flatten the list of batches and return an iterator

    def __len__(self):
        return len(self.lengths)

# Collate function

def collate_fn(batch, pad_idx):
    src_batch = [example["src_ids"] for example in batch]
    # print(src_batch[0:2])
    trg_batch = [example["trg_ids"] for example in batch]
    # print(trg_batch[0:2])

    def pad(sequences):
        max_len = max(len(seq) for seq in sequences) #获取最长句子的长度
        return [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences] #将每个句子补齐到相同的长度

    src_padded = pad(src_batch) #在短句子后面用1补充到batch内句子最长长度
    # print(src_padded[0:2])
    # print(len(src_padded))
    trg_padded = pad(trg_batch)

    # 创建 Batch 对象
    return Batch(torch.tensor(src_padded, dtype=torch.long),
                 torch.tensor(trg_padded, dtype=torch.long),
                 pad_idx)
# Load and prepare

def prepare_data(batch_size=64, small_data=True):
    raw_data = load_dataset("iwslt2017", "iwslt2017-de-en", split={"train": "train", "validation": "validation", "test": "test"})

    if small_data:
        raw_data["train"] = raw_data["train"].select(range(100))
        raw_data["validation"] = raw_data["validation"].select(range(20))
        raw_data["test"] = raw_data["test"].select(range(20))
    
    train_data = list(filter(filter_none, map(preprocess, raw_data["train"]["translation"])))
    val_data = list(filter(filter_none, map(preprocess, raw_data["validation"]["translation"])))
    test_data = list(filter(filter_none, map(preprocess, raw_data["test"]["translation"])))

    # print(train_data[0:2])

    src_vocab = build_vocab(train_data, "src")
    tgt_vocab = build_vocab(train_data, "trg")

    # print(src_vocab.itos)
    # print(src_vocab.stoi)

    # print(train_data[0:2])

    train_data = list(map(lambda ex: encode(ex, src_vocab, tgt_vocab), train_data))
    val_data = list(map(lambda ex: encode(ex, src_vocab, tgt_vocab), val_data))
    test_data = list(map(lambda ex: encode(ex, src_vocab, tgt_vocab), test_data))

    # print(train_data[0:2])
    
    """     
    [{'src': ['Vielen', 'Dank', ',', 'Chris', '.']
    , 'trg': ['<s>', 'Thank', 'you', 'so', 'much', ',', 'Chris', '.', '</s>'],
     'src_ids': [0, 4, 5, 6, 7], 'trg_ids': [2, 0, 4, 5, 6, 7, 8, 9, 3]}
    , {'src': ['Es', 'ist', 'mir', 'wirklich', 'eine', 'Ehre', ',', 'zweimal', 'auf', 'dieser', 'Bühne', 'stehen', 'zu', 'dürfen', '.', 'Tausend', 'Dank', 'dafür', '.']
    , 'trg': ['<s>', 'And', 'it', "'s", 'truly', 'a', 'great', 'honor', 'to', 'have', 'the', 'opportunity', 'to', 'come', 'to', 'this', 'stage', 'twice', ';', 'I', "'m", 'extremely', 'grateful', '.', '</s>']
    , 'src_ids': [8, 9, 10, 11, 12, 0, 5, 0, 13, 14, 0, 0, 15, 0, 7, 0, 4, 16, 7]
    , 'trg_ids': [2, 10, 11, 12, 0, 13, 0, 0, 14, 15, 16, 17, 14, 18, 14, 19, 0, 0, 20, 21, 22, 23, 0, 9, 3]}] 
    """

    pad_idx = src_vocab[BLANK_WORD] #0
    # print(f"Padding index: {pad_idx}")
    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    test_dataset = TranslationDataset(test_data)

    lengths = [len(example["src_ids"]) for example in train_data]#每个句子的长度的列表
    sampler = SortedBatchSampler(lengths, batch_size=batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: collate_fn(x, pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx))

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab