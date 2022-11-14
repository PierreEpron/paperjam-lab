from collections import Counter
import random
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import functools
import itertools

# NER 

def ents_to_iob(words, ner):
    iob = ['O'] * len(words)
    for ent_start, ent_end, ent_label in ner:
        iob[ent_start] = f'B-{ent_label}'
        for i in range(ent_start+1, ent_end):
            iob[i] = f'I-{ent_label}'
    return iob

def labels_to_id(data):
    labels = [[ent_label for _, _, ent_label in item['ner']] for item in data]
    labels = functools.reduce(lambda a, b : a + b, labels)
    ids = {'O': 0}
    for label in set(labels):
        ids.update({f'B-{label}':len(ids)})
        ids.update({f'I-{label}':len(ids)})
    return ids

def data_to_ner(data):
    sents = []
    for item in data:
        words = item['words']
        ner = item['ner']
        iob = ents_to_iob(words, ner)
        for sent_start, sent_end in item['sentences']:
            sents.append({'tokens': words[sent_start:sent_end], 'tags': iob[sent_start:sent_end]})
    return sents

# Coref

def generate_pairs(data):
    pairs = []
    for item in data:
        # create Span : Label clusters 
        # TODO : information on multiple label by span
        clusters = {}
        for k, vlist in item["coref"].items():
            for v in vlist:
                if tuple(v) not in clusters:
                    clusters[tuple(v)] = []
                clusters[tuple(v)].append(k)

        clusters = {k: set(v) for k, v in clusters.items()}
        
        # print('\n'.join([f'{k} : {v} : {item["words"][k[0]:k[1]]}' for k, v in clusters.items() if len(v) > 1]))
        
        # entities = [tuple(x) for x in ]
        for ent_1, ent_2 in itertools.combinations(item["ner"], 2):
            
            # unpack ents
            start_1, end_1, type_1 = ent_1
            start_2, end_2, type_2 = ent_2

            # if ents share type continue 
            if type_1 != type_2:
                continue

            # retrieve cluster labels associate to ents, empty set if ents not found
            cluster_labels_1 = clusters.get((start_1, end_1), set())
            cluster_labels_2 = clusters.get((start_2, end_2), set())

            # ents words 
            w1 = item["words"][start_1:end_1]
            w2 = item["words"][start_2:end_2]

            # by default b-classification is negatif
            gold_label = 0

            # if ents strings are the same or if ents share a cluster label : b-classification is positif 
            if " ".join(w1).lower() == " ".join(w2).lower() or len(cluster_labels_1 & cluster_labels_2) > 0:
                gold_label = 1
            # if ents has no cluster label classification : skip
            elif len(cluster_labels_1) == 0 and len(cluster_labels_2) == 0:
                continue

            # TODO : We should be sure we want handle words equality in training

            pairs.append(((type_1, w1, type_2, w2, gold_label), (start_1, end_1), (start_2, end_2)))
    return pairs

def compute_prob(pairs):
    c = Counter([x[-1] for x, _, _ in pairs])
    min_count = min(c.values())
    prob = {k: min(1, min_count / v) for k, v in c.items()}
    return prob

class BaseDataLoader(object):
    
    def __init__(self):
        pass
    
    def collate_fn(self, batch_as_list):
        return {}
    
    def create_dataloader(self, data, batch_size=1, num_workers=1, **kwgs):
        """
        Create a torch dataloader

        Args:
            data (list[dict]): data to load.
            collate_fn (callable): collate_fn for DataLoader.
            batch_size (int): batch_size for DataLoader (default=1)
            num_workers (int): num_workers for DataLoader (default=1).

        Returns:
            Torch dataloader
        """
        return DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn, **kwgs)

class NERDataLoader(BaseDataLoader):

    def __init__(self, tokenizer, label_ids={}):
        self.tokenizer = tokenizer
        self.label_ids = label_ids
        super().__init__()

    def tokenize_and_align_labels(self, sample):
        """
        Align tokens and labels

        Args:
            sample (dict): a data sample with tokens and tags.

        Returns:
            dict
        """

        tokens = sample['tokens']

        # for prediction when labels are missing
        if 'tags' in sample:
            tags = sample['tags']
        else:
            tags = len(tokens) * ['O']

        encoded_sentence = []
        aligned_labels = []
        word_label = []
        iob_labels = []

        for t, n in zip(tokens, tags):

            encoded_token = self.tokenizer.tokenize(t)

            if len(encoded_token) < 1:
                encoded_token = [self.tokenizer.unk_token]

            encoded_token = self.tokenizer.convert_tokens_to_ids(encoded_token)

            n_subwords = len(encoded_token)

            if len(encoded_sentence) + len(encoded_token) > 512:
                break

            encoded_sentence.extend(encoded_token)

            aligned_labels.extend(
                [self.label_ids[n]] + (n_subwords - 1) * [-1]
            )

            word_label.append(self.label_ids[n])
            iob_labels.append(n)

        assert len(encoded_sentence) == len(aligned_labels)

        encoded_sentence = torch.LongTensor(encoded_sentence)
        aligned_labels = torch.LongTensor(aligned_labels)
        word_label = torch.LongTensor(word_label)

        lengths = len(iob_labels)

        return {
            'input_ids': encoded_sentence, 'aligned_labels': aligned_labels,
            'true': iob_labels, 'seq_length': lengths, 'word_label': word_label
        }

    def collate_fn(self, batch_as_list):
        """
        Batchification

        Args:
            batch_as_list (list[dict]): a list of dict with "tokens" and "tags"

        Returns:
            Input tensors for model
        """

        batch = [self.tokenize_and_align_labels(b) for b in batch_as_list]

        input_ids = pad_sequence([b['input_ids'] for b in batch],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)

        aligned_labels = pad_sequence(
            [b['aligned_labels'] for b in batch], batch_first=True, padding_value=-1)

        iob_labels = [b['true'] for b in batch]

        word_label = pad_sequence(
            [b['word_label'] for b in batch], batch_first=True, padding_value=0)

        seq_length = [b['seq_length'] for b in batch]

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        subword_mask = aligned_labels != -1

        return {
            'input_ids': input_ids,
            'aligned_labels': aligned_labels,
            'attention_mask': attention_mask,
            'seq_length': seq_length,
            'subword_mask': subword_mask,
            'word_label': word_label,
            'true': iob_labels
        }

    def create_dataloader(self, data, is_train=False, batch_size=1, num_workers=1, **kwgs):
        ner = data_to_ner(data)
        self.label_ids = labels_to_id(data)
        return super().create_dataloader(ner, batch_size, num_workers, **kwgs)

class CorefDataLoader(BaseDataLoader):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        super().__init__()

    def tokenize(self, sample):

        sample, span_1, span_2 = sample
        t1, w1, t2, w2, gold_label = sample

        tokens = [self.tokenizer.cls_token, t1] + w1 + [self.tokenizer.sep_token, t2] +  w2
        encoded_sentence = []
            
        for t in tokens:

            encoded_token = self.tokenizer.tokenize(t)
            
            if len(encoded_token) < 1:
                encoded_token = [self.tokenizer.unk_token]

            encoded_token = self.tokenizer.convert_tokens_to_ids(encoded_token)

            if len(encoded_sentence) + len(encoded_token) > 512:
                break
                # TODO : Warnings with a verbose args ?

            encoded_sentence.extend(encoded_token)

        return {
            'input_ids': torch.LongTensor(encoded_sentence),
            'true': gold_label,
            "metadata": {
                "span_1": span_1,
                "span_2": span_2,
            }
        }

    def collate_fn(self, batch_as_list):
        """
        Batchification

        Args:
            batch_as_list (list[dict]): ...

        Returns:
            Input tensors for model
        """
        
        batch = [self.tokenize(b) for b in batch_as_list]
        
        input_ids = pad_sequence([b['input_ids'] for b in batch],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        metadata = [b['metadata'] for b in batch]
        true = [b['true'] for b in batch]

        return {
            "tokens": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata,
            "true": torch.IntTensor(true)
        }

    def create_dataloader(self, data, is_train=False, batch_size=1, num_workers=1, **kwgs):
        pairs = generate_pairs(data)
        if is_train:
            prob = compute_prob(pairs)
            print(len(pairs))
            random.seed(42)
            pairs = [pair for pair, _, _ in pairs if random.random() < prob[pair[-1]]]
            print(len(pairs))

        return super().create_dataloader(pairs, batch_size, num_workers, **kwgs)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from helpers import read_jsonl

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    data = read_jsonl('data/train.jsonl')

    # Coref
    loader = CorefDataLoader(tokenizer).create_dataloader(data, is_train=True, prefetch_factor=1)

    # NER 
    # loader = NERDataLoader(tokenizer).create_dataloader(data, prefetch_factor=1)

    for b in loader:
        print(b)
        break     