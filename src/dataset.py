import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def create_dataloader(data, collate_fn, batch_size=1, num_workers=1, **kwgs):
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
    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **kwgs)

def ents_to_iob(words, ner):
    iob = ['O'] * len(words)
    for ent_start, ent_end, ent_label in ner:
        iob[ent_start] = f'B-{ent_label}'
        for i in range(ent_start+1, ent_end):
            iob[i] = f'I-{ent_label}'
    return iob

def labels_to_id(labels):
    ids = {'O': 0}
    for label in set(labels):
        ids.update({f'B-{label}':len(ids)})
        ids.update({f'I-{label}':len(ids)})
    return ids

def data_to_ner(data):
    sents = []
    labels = []
    for item in data:
        words = item['words']
        ner = item['ner']
        iob = ents_to_iob(words, ner)
        labels += [ent_label for _, _, ent_label in ner]
        for sent_start, sent_end in item['sentences']:
            sents.append({'tokens': words[sent_start:sent_end], 'tags': iob[sent_start:sent_end]})
    return labels_to_id(labels), sents

class BaseDataLoader(object):
    
    def __init__(self):
        pass
    
    def collate_fn(self, batch_as_list):
        return {}
    
    def create_dataloader(self, data, batch_size=1, num_workers=1, **kwgs):
        return create_dataloader(data, self.collate_fn, batch_size=batch_size, num_workers=num_workers, **kwgs)

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
            'iob_labels': iob_labels, 'seq_length': lengths, 'word_label': word_label
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

        iob_labels = [b['iob_labels'] for b in batch]

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
            'iob_labels': iob_labels
        }

    def create_dataloader(self, data, batch_size=1, num_workers=1, **kwgs):
        label_ids, ner = data_to_ner(data)
        self.label_ids = label_ids
        return super().create_dataloader(ner, batch_size, num_workers, **kwgs)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from helpers import read_jsonl

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    data = read_jsonl('data/train.jsonl')
    loader = NERDataLoader(tokenizer).create_dataloader(data, prefetch_factor=1)
    
    for b in loader:
        print(b)
        break