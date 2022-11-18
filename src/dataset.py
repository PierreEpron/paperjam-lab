from collections import Counter
import random
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import functools
import itertools

import numpy as np

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

# Relation Extraction 

experiment_words_to_check = set("experiment|evaluation|evaluate|evaluate".split("|"))
dataset_words_to_check = set("dataset|corpus|corpora".split("|"))

def get_ent_type(span, doc):
    ''' Find entity span in document ner data for given span'''
    span_start, span_end = span
    for ent in doc['ner']:
        if span_start == ent[0] and span_end == ent[1]:
            return ent[2]
    raise RuntimeError(f"No entity type found for {span} in {doc['doc_id']}")

def is_span_inside(span, other):
    return span[0] >= other[0] and span[1] <= other[1]

def get_span_words(span, doc):
    return doc['words'][span[0]:span[1]]

def get_section_features(doc):
    features_list = []
    for i, (s, e) in enumerate(doc['sections']):
        features = []
        words = " ".join(doc['words'][s:e]).lower()
        if i == 0:
            features.append("Heading")

        if "abstract" in words:
            features.append("Abstract")
        if "introduction" in words:
            features.append("Introduction")

        if any(w in words for w in dataset_words_to_check):
            features.append("Dataset")
            features.append("Experiment")

        if any(w in words for w in experiment_words_to_check):
            features.append("Experiment")

        features_list.append(sorted(list(set(features))))

    return features_list


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
        golds = []

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
            golds.append(n)

        assert len(encoded_sentence) == len(aligned_labels)

        encoded_sentence = torch.LongTensor(encoded_sentence)
        aligned_labels = torch.LongTensor(aligned_labels)
        word_label = torch.LongTensor(word_label)

        lengths = len(golds)

        return {
            'input_ids': encoded_sentence, 'aligned_labels': aligned_labels,
            'golds': golds, 'seq_length': lengths, 'word_label': word_label
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

        golds = [b['golds'] for b in batch]

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
            'golds': golds
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
            'gold_label': gold_label,
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
        golds = [b['gold_label'] for b in batch]

        return {
            "tokens": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata,
            "golds": golds
        }

    def create_dataloader(self, data, is_train=False, batch_size=1, num_workers=1, **kwgs):
        pairs = generate_pairs(data)
        if is_train:
            prob = compute_prob(pairs)
            random.seed(42)
            pairs = [pair for pair in pairs if random.random() < prob[pair[0][-1]]]

        return super().create_dataloader(pairs, batch_size, num_workers, **kwgs)

class RelDataLoader(BaseDataLoader):
    def __init__(self, tokenizer, labels=None, max_paragraph_len=300):
        self.tokenizer = tokenizer
        self.max_paragraph_len = max_paragraph_len
        self.labels = labels if labels else ['Task', 'Material', 'Metric', 'Method']
        self.label_idx_map = {l:i for i, l in enumerate(self.labels)}
        self.idx_label_map = {i:l for i, l in enumerate(self.labels)}

        self.sf_labels = ['Abstract', 'Dataset', 'Experiment', 'Heading', 'Introduction']
        self.sf_idx_map = {l:i for i, l in enumerate(self.sf_labels)}
        self.idx_sf_map = {i:l for i, l in enumerate(self.sf_labels)}
        super().__init__()    

    def label_onehot(self, ids, idx_label_map):
        return [1 if i in ids else 0 for i in range(len(idx_label_map))]

    def chunk_sentence(self, start, end):
        return [[i, min(i+self.max_paragraph_len, end)] for i in range(start, end, self.max_paragraph_len)]

    def section_to_paragraphs(self, section_span, doc):

        # retrieve all section sentences        
        sentence_spans = [sentence for sentence in doc['sentences'] if is_span_inside(sentence, section_span)]

        paragraph_spans = []
        # start paragraph with first sentence of section
        paragraph_start, paragraph_end = sentence_spans[0]
        for sentence_start, sentence_end in sentence_spans[1:]:
            # if current paragraph is longer than self.max_paragraph_len
            # append it into paragraph_spans and start a new paragraph
            if sentence_end - paragraph_start > self.max_paragraph_len:
                paragraph_spans.extend(self.chunk_sentence(paragraph_start, paragraph_end))
                paragraph_start = sentence_start
            paragraph_end = sentence_end
        # append the last paragraph of the section
       
        # handle case of first sentence is larger self.max_paragraph_len
        paragraph_spans.extend(self.chunk_sentence(paragraph_start, paragraph_end))
        
        # make sure pragraphs length equals to section length
        assert section_span[1] - section_span[0] == paragraph_spans[-1][1] - paragraph_spans[0][0], \
            f"section length  ({section_span}, {section_span[1] - section_span[0]}) don't match paragraph length ([{paragraph_spans[0][0]}, {paragraph_spans[-1][1]}], {paragraph_spans[-1][1] - paragraph_spans[0][0]})"

        return paragraph_spans

    def doc_to_paragraphs(self, doc):

        doc_id = doc['doc_id']
        coref = {k:v for k, v in doc['coref'].items() if len(v) > 0}
        words = doc['words']
        n_ary_relations = doc['n_ary_relations']

        ccluster_idx_map = {l:i for i, l in enumerate(coref)}
        idx_ccluster_map = {i:l for i, l in enumerate(coref)}
        ccluster_label_map = {ccluster_idx_map[k]:self.label_idx_map[get_ent_type(v[0], doc)] for k, v in coref.items() if len(v) > 0}

        # I should make entity_to_clusters here
        relation_to_cluster_ids = {}
        for rel_idx, rel in enumerate(n_ary_relations):
            relation_to_cluster_ids[rel_idx] = []
            for entity in self.labels:
                if rel[entity] in ccluster_idx_map:
                    relation_to_cluster_ids[rel_idx].append(ccluster_idx_map[rel[entity]])

        section_features = get_section_features(doc)
        refs = functools.reduce(lambda a, b : a  + b, coref.values()) if len(coref) > 0 else []

        def span_has_ref(span):
            ''' Return true if there is at least 1 mention from 1 coref in span'''
            for ref_span in refs:
                if is_span_inside(ref_span, span):
                    return True
            return False

        # break section into paragraphs 
        # TODO : Handle sentences longer than self.max_paragraph_len
        paragraph_spans = []
        paragraph_sf = []
        for section_span, sf in zip(doc['sections'], section_features):
            p = self.section_to_paragraphs(section_span, doc)
            paragraph_spans.extend(p)
            for _ in p:
                paragraph_sf.append([self.sf_idx_map[sfl] for sfl in sf])

        # ATM remove paragraph without any mention
        # paragraph_spans = [p for p in paragraph_spans if span_has_ref(p)]
        # paragraph_sf = [sf for p, sf in zip(paragraph_spans, paragraph_sf) if span_has_ref(p)]

        coref_spans = []
        coref_spans_type_map = []
        coref_sf_map = []
        coref_cluster_map = []

        entity_to_clusters = {k:[] for k in self.idx_label_map}

        for paragraph_span, sf in zip(paragraph_spans, paragraph_sf):
            coref_spans.append([])
            coref_spans_type_map.append([])
            coref_sf_map.append([])
            coref_cluster_map.append([])
            for k, listv in coref.items():
                ent_type = self.label_idx_map[get_ent_type(listv[0], doc)]
                for v in listv:
                    if is_span_inside(v, paragraph_span):
                        coref_spans[-1].append(v)
                        coref_spans_type_map[-1].append(ent_type)
                        coref_sf_map[-1].append(sf)
                        coref_cluster_map[-1].append(ccluster_idx_map[k])
                entity_to_clusters[ent_type].append(ccluster_idx_map[k])
        
        entity_to_clusters = {k:list(set(v)) for k, v in entity_to_clusters.items()}

        # print('ccluster_idx_map : ', ccluster_idx_map)
        # print('entity_to_clusters : ', entity_to_clusters)

        return {
            'doc_id':doc_id,
            'coref':coref,
            'words':words,
            
            'ccluster_idx_map':ccluster_idx_map,
            'idx_ccluster_map':idx_ccluster_map,
            'ccluster_label_map':ccluster_label_map,
            'entity_to_clusters':entity_to_clusters,
            'relation_to_cluster_ids':relation_to_cluster_ids,
            
            'paragraph_spans':paragraph_spans,
            'coref_spans':coref_spans,
            'coref_spans_type_map':coref_spans_type_map,
            'coref_sf_map':coref_sf_map,
            'coref_cluster_map':coref_cluster_map
        }   

    def tokenize_word(self, word):
        token = self.tokenizer.tokenize([word], is_split_into_words=True)
        
        if len(token) < 1:
            token = [self.tokenizer.unk_token]

        return self.tokenizer.convert_tokens_to_ids(token)

    # def tokenize_sentence(self, sentence, doc):
    #     return functools.reduce(
    #         lambda a, b : a + b, 
    #         [self.tokenize_word(word) for word in get_span_words(sentence, doc)])

    def tokenize_paragraph(self, paragraph, doc):

        tokens = [self.tokenize_word(word) for word in get_span_words(paragraph, doc)]
        tokens = functools.reduce(lambda a, b : a  + b, tokens)

        if len(tokens) > 512:
            print(f"MAXIMUM TOKEN SIZE ({len(tokens)}/512) REACHED FOR DOC : {doc['doc_id']}")
            tokens = tokens[:512]

        return tokens

    def collate_fn(self, batch_as_list):
        b = batch_as_list[0]
 
        ccluster_idx_map, idx_ccluster_map, ccluster_label_map = (b['ccluster_idx_map'], b['idx_ccluster_map'], b['ccluster_label_map'])
        relation_to_cluster_ids = b['relation_to_cluster_ids']
            

        # Tokenize words
        tokens = [torch.LongTensor(self.tokenize_paragraph(paragraph_spans, b)) for paragraph_spans in b['paragraph_spans']]

        # Pad tokens
        input_ids = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # print('input_ids.shape : ', input_ids.shape)

        # Mask padded tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        # print('attention_mask.shape : ', attention_mask.shape)
        
        # Make coref spans inclusif and relatif to paragraph
        # Also make coref spans mean position
        # Also make coref label onehot

        doc_length = len(b['words'])
        
        relatif_spans = []
        spans_position = []
        spans_label_onehot = []
        spans_sf_onehot = []
        spans_ccluster_onehot = []

        for coref_spans, paragraph_spans, coref_spans_type_map, coref_sf_map, coref_cluster_map  in zip(
            b['coref_spans'], 
            b['paragraph_spans'], 
            b['coref_spans_type_map'], 
            b['coref_sf_map'],
            b['coref_cluster_map']):

            paragraph_start, _ = paragraph_spans
            
            relatif_spans.append([])
            spans_position.append([])
            spans_label_onehot.append([])
            spans_sf_onehot.append([])
            spans_ccluster_onehot.append([])

            for coref_span, coref_span_type, coref_sf, coref_cluster in zip(coref_spans, coref_spans_type_map, coref_sf_map, coref_cluster_map):
                coref_start, coref_end = coref_span
                relatif_spans[-1].append([coref_start-paragraph_start, coref_end-paragraph_start-1])
                spans_position[-1].append(np.mean([coref_start, coref_end]) / doc_length)
                spans_label_onehot[-1].append(self.label_onehot([coref_span_type], self.idx_label_map))
                spans_sf_onehot[-1].append(self.label_onehot(coref_sf, self.idx_sf_map))
                spans_ccluster_onehot[-1].append(self.label_onehot([coref_cluster], idx_ccluster_map))

        # Pad relatif_spans & spans_position & spans_label_onehot & spans_sf_onehot
        max_span_count = max([len(s) for s in relatif_spans])

        for spans, positions, label_onehot, sf_onehot, ccluster_onehot in zip(relatif_spans, spans_position, spans_label_onehot, spans_sf_onehot, spans_ccluster_onehot):
            for i in range(len(spans), max_span_count):
                spans.append([0,0])
                positions.append(0)
                label_onehot.append(self.label_onehot([], self.idx_label_map))
                sf_onehot.append(self.label_onehot([], self.idx_sf_map))
                ccluster_onehot.append(self.label_onehot([], idx_ccluster_map))

        relatif_spans = torch.LongTensor(relatif_spans)
        spans_position = torch.FloatTensor(spans_position).unsqueeze(-1)
        spans_label_onehot = torch.FloatTensor(spans_label_onehot)
        spans_sf_onehot = torch.FloatTensor(spans_sf_onehot)
        spans_ccluster_onehot = torch.FloatTensor(spans_ccluster_onehot)

        # print('relatif_spans.shape : ', relatif_spans.shape)
        # print('spans_position.shape : ', spans_position.shape)
        # print('spans_label_onehot.shape : ', spans_label_onehot.shape)
        # print('spans_sf_onehot.shape : ', spans_sf_onehot.shape)
        # print('coref_cluster_onehot.shape : ', coref_cluster_onehot.shape)

        # Mask padded spans
        spans_mask = (relatif_spans != 0).sum(-1).bool()
        # print('spans_mask.shape : ', spans_mask.shape)
        
        # print('paragraph_spans.len : ', len(b['paragraph_spans']))
        # print('ccluster_label_map.len : ', len(ccluster_label_map))

        return {
            'doc_id' : b['doc_id'],
            'cluster_to_type_arr': [t for _, t in sorted(ccluster_label_map.items())],
            'entity_to_clusters':b['entity_to_clusters'],
            'relation_to_cluster_ids':relation_to_cluster_ids,
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'coref_spans':relatif_spans,
            'coref_spans_mask':spans_mask,
            'coref_spans_position': spans_position, # (B, P, 1)
            'coref_spans_labels': spans_label_onehot, # (B, P, L)
            'coref_spans_sf':spans_sf_onehot, # (B, S, SF)
            'coref_spans_cluster':spans_ccluster_onehot, # (B, S, C)
        }

    def create_dataloader(self, data, is_train=False, batch_size=1, num_workers=1, **kwgs):
        docs = [self.doc_to_paragraphs(doc) for doc in data]
        return super().create_dataloader(docs, batch_size, num_workers, **kwgs)
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from helpers import read_jsonl

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    data = read_jsonl('data/train.jsonl')

    # Relation

    loader = RelDataLoader(tokenizer).create_dataloader(data, batch_size=1, prefetch_factor=1)

    for b in loader:
        print(b)
        # from pathlib import Path
        # import json
        # Path('a.json').write_text(json.dumps({'coref':b['coref'], 'paragraph_spans':b['paragraph_spans'], 'coref_sf':b['coref_sf']}))
        break

    # Coref
    # loader = CorefDataLoader(tokenizer).create_dataloader(data, is_train=True, prefetch_factor=1)

    # NER 
    # loader = NERDataLoader(tokenizer).create_dataloader(data, prefetch_factor=1)

   