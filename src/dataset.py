from collections import Counter, defaultdict
import random
from typing import Any, Dict, Iterable, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import functools
import itertools

import numpy as np
from tqdm import tqdm
import transformers

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
            sents.append({'doc_id':item['doc_id'], 'tokens': words[sent_start:sent_end], 'tags': iob[sent_start:sent_end]})
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

            pairs.append((item['doc_id'], (type_1, w1, type_2, w2, gold_label), (start_1, end_1), (start_2, end_2)))
    return pairs

def compute_prob(pairs):
    c = Counter([x[-1] for _, x, _, _ in pairs])
    min_count = min(c.values())
    prob = {k: min(1, min_count / v) for k, v in c.items()}
    return prob

# Relation Extraction 

EXPERIMENT_WORDS_TO_CHECK = set("experiment|evaluation|evaluate|evaluate".split("|"))
DATASET_WORDS_TO_CHECK = set("dataset|corpus|corpora".split("|"))


def is_span_inside(span, other):
    return span[0] >= other[0] and span[1] <= other[1]

def get_span_words(span, doc):
    return doc['words'][span[0]:span[1]]

class BaseDataLoader(object):
    
    def __init__(self):
        pass
    
    def get_ent_type(span : Tuple[int, int], doc : Dict[str, Any]) -> str:
        """
            Find entity type in `doc['ner']` value for given `span`.

            Parameters
            ----------
            span : `Tuple[int, int]`
                Target entity start and end.
            doc : `Dict[str, Any]`
                Dictionary with a 'ner' key associated to a list of entity tuple.

            Returns
            -------
            entity_type : `str`
                Return entity type associated to given `span`.
        """
        span_start, span_end = span
        for ent in doc['ner']:
            if span_start == ent[0] and span_end == ent[1]:
                return ent[2]
        raise RuntimeError(f"No entity type found for {span} in {doc['doc_id']}")

    def get_section_features(section_span : Tuple[int, int], section_idx : int, doc : Dict[str, Any]) -> List[str]:
        """
            Get section features in `doc['words']` values for given `section_span`.

            Parameters
            ----------
            section_span : `Tuple[int, int]`
                Span of target section.
            section_idx : `int`
                Index of section in doc. Use to add `Heading` feature
            doc : `Dict[str, Any]`
                Dictionary with a 'words' key associated to a list of str.

            Returns
            -------
            section_features : `str`
                The `section_features` list found in `doc['words']` by using `section_span`.
        """
        # TODO : Test

        start, end = section_span
        words = " ".join(doc['words'][start:end]).lower()

        features = []

        if section_idx == 0:
            features.append("Heading")

        if "abstract" in words:
            features.append("Abstract")
        if "introduction" in words:
            features.append("Introduction")

        if any(w in words for w in DATASET_WORDS_TO_CHECK):
            features.append("Dataset")
            features.append("Experiment")

        if any(w in words for w in EXPERIMENT_WORDS_TO_CHECK):
            features.append("Experiment")

        return features

    def tokenize_word(self, word : str) -> List[int]:
        """
            Retrieve words in given `doc` and perform documentation on thoses words.

            Parameters
            ----------
            span : `Tuple[int, int]`
                Start and end of sequence to tokenize.

            Returns
            -------
            tokenised_word: `List[int]`
                Ids of tokens of the given `word`.
        """

        token = self.tokenizer.tokenize([word], is_split_into_words=True)
        
        if len(token) < 1:
            token = [self.tokenizer.unk_token]

        return self.tokenizer.convert_tokens_to_ids(token)

    def tokenize_span(self, span : Tuple[int, int], doc : Dict[str, Any]) -> List[int]:
        """
            Retrieve words in given `doc` and perform documentation on thoses words.

            Parameters
            ----------
            span : `Tuple[int, int]`
                Start and end of sequence to tokenize.
            doc : `Dict[str, Any]`
                Dictionary with a `words` key associated to a list str and `doc_id` key any id used for this `doc`.

            Returns
            -------
            tokenised_span : `List[int]`
                Ids of tokens of the given `span`.
        """

        tokens = [self.tokenize_word(word) for word in get_span_words(span, doc)]
        tokens = functools.reduce(lambda a, b : a  + b, tokens)
        assert len(tokens) <= 512, f"MAXIMUM TOKEN SIZE ({len(tokens)}/512) REACHED FOR DOC : {doc['doc_id']}"

        return tokens

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
            'doc_id':sample['doc_id'],
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
            'doc_id':[b['doc_id'] for b in batch], 
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
        doc_id, sample, span_1, span_2 = sample
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
            'doc_id': doc_id,
            'input_ids': torch.LongTensor(encoded_sentence),
            'gold_label': gold_label,
            'word_1': w1,
            'word_2': w2,
            "span_1": span_1,
            "span_2": span_2,
            "type_1": t1,
            "type_2": t2
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
        
        golds = [b['gold_label'] for b in batch]

        return {
            'doc_id': [b['doc_id'] for b in batch],
            "tokens": input_ids,
            "attention_mask": attention_mask,
            "golds": golds,
            'word_1': [b['word_1'] for b in batch],
            'word_2': [b['word_2'] for b in batch],
            'span_1': [b['span_1'] for b in batch],
            'span_2': [b['span_2'] for b in batch],
            'type_1': [b['type_1'] for b in batch],
            'type_2': [b['type_2'] for b in batch],
        }

    def create_dataloader(self, data, is_train=False, batch_size=1, num_workers=1, **kwgs):
        pairs = generate_pairs(data)
        if is_train:
            prob = compute_prob(pairs)
            random.seed(42)
            pairs = [pair for pair in pairs if random.random() < prob[pair[1][-1]]]

        return super().create_dataloader(pairs, batch_size, num_workers, **kwgs)

class RelDataLoader(BaseDataLoader):
    """
        Loader for Relation Classification Model.

        Parameters
        ----------
        tokenizer : `transformers.AutoTokenizer`
            Tokenizer used for preprocess data.
        entities : `List[str]`, default=None 
            Entity types that will be handled by this model 
            if `None` assign ['Task', 'Material', 'Metric', 'Method'] by default.
        sfeatures : `List[str]`, default=None 
            Section Feature types that will be handled by this model 
            if `None` assign  ['Abstract', 'Dataset', 'Experiment', 'Heading', 'Introduction'] by default.
        max_paragraph_len : `int`, default=300
            Maximum allowed paragraph len, all paragraph with length > `max_paragraph_len` will be truncated.
    """
    def __init__(
        self, 
        tokenizer : transformers.AutoTokenizer, 
        entities: List[str] = None, 
        sfeatures: List[str] = None, 
        max_paragraph_len: int = 300,
        max_relation_len: int = 64
        ):
        self.tokenizer = tokenizer
        self.max_paragraph_len = max_paragraph_len
        self.max_relation_len = max_relation_len

        self.entities = entities if entities else ['Task', 'Material', 'Metric', 'Method']
        self.sfeatures = sfeatures if sfeatures else ['Abstract', 'Dataset', 'Experiment', 'Heading', 'Introduction']

        # Map entities with ids
        self.entity_to_idx, self.idx_to_entity = RelDataLoader.get_map_lists(self.entities)

        # Map sfeatures with ids
        self.sfeature_to_idx, self.idx_to_sfeature = RelDataLoader.get_map_lists(self.sfeatures)

        super().__init__()    

    def get_map_lists(list_1 : List[Any], list_2 : List[Any] = None) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """
            Return mapping dicts for list_1 to list_2 and list_2 to list_1.
            if list_2 is None assign ids of list_1.

            Parameters
            ----------
            list_1 : `List[Any]`
                First list to map.
            list_2 : `List[Any]`, default=None
                Second list to map.
                if None assign `range(len(list_1))`

            Returns
            -------
            list_1_to_list_2 : `Dict[Any, Any]`
                Map items from list_1 to items from list_2.
            list_2_to_list_1 : `Dict[Any, Any]`
                Map items from list_2 to items from list_1.
        """

        list_2 = list_2 if list_2 else list(range(len(list_1)))

        return {l1:l2 for l1, l2 in zip(list_1, list_2)}, {l2:l1 for l1, l2 in zip(list_1, list_2)}

    def filter_relations(relations : Iterable[Dict[str, str]], coref_keys : Iterable[str]):
        """
        Filter relations in `doc` with valid coref keys.

        Parameters
        ----------
        doc : `Iterable[Dict[str, str]]`
            List of relations. Example {'Task':'A', 'Material':'B', Dataset:'C', Method':'D'}
        corefs : `Iterable[str]`
            List of coref valid keys used to  filter

        Returns
        -------
        filtered_relations : `List[Dict[str, str]]` 
            Filtered list of relations.
        """
        return [relation for relation in relations if set(relation.values()).intersection(coref_keys)]

    def get_onehot(one_ids : List[int], full_ids : List[int]) -> List[int]:
        """
            Return onehot encoded vector of `one_ids` in `full_ids`.

            Parameters
            ----------
            one_ids : `List[int]`
                One ids of the onehot vector.
            full_ids : str, List[int]`
                Full ids of the onehot vector.

            Returns
            -------
            onehot_vector : `List[int]`
                One of vector of `one_ids` in `full_ids`.
        """
        return [1 if i in one_ids else 0 for i in range(len(full_ids))]

    def increase_span_boundaries(spans, idx, n):
        '''add n to span starts or ends if they are greater than idx'''
        return [[s if s <= idx else s + n, e if e <= idx else e + n] for s, e in spans]
    
    def tokenize_section(
        self,
        section : Tuple[int, int], 
        sentences : Iterable[Tuple[int, int]], 
        corefs : Dict[str, Tuple[int,int]],
        words : Iterable[str]
    ) -> Dict[str, Any]:

        s_start, s_end = section

        # Filter sentences & corefs wich are in section
        sentences = [[s, e] for s, e in sentences if is_span_inside((s, e), section)]
        corefs = {k:[v for v in listv if is_span_inside(v, section)] for k, listv in corefs.items()}
        corefs = {k:listv for k, listv in corefs.items() if len(listv) > 0}
        tokenized_words = []

        for i, w in enumerate(words[s_start:s_end]):
            idx = s_start + i
            tokenized_word = self.tokenize_word(w)
            if len(tokenized_word) > 1:
                n = len(tokenized_word) - 1
                sentences = RelDataLoader.increase_span_boundaries(sentences, idx, n)
                for listv in corefs.values():
                    listv = RelDataLoader.increase_span_boundaries(listv, idx, n)

            tokenized_words.extend(tokenized_word)

        return tokenized_words, sentences, corefs

    def chunk_section(self, corefs, sentences):
        mentions = [] if len(corefs) == 0 else functools.reduce(lambda a, b: a + b, [listv for listv in corefs.values()])            
        sentences = self.chunk_sentences(list(sentences), mentions)
        paragraphs = self.sentences_to_paragraphs(sentences)
        return sentences, paragraphs

    def chunk_sentences(self, sentences : Iterable[Tuple[int, int]], mentions : Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # TODO : Docstring & Test
        
        valid_sentences = []
        
        while len(sentences) > 0:
            s_start, s_end = sentences.pop(0)
            size = s_end - s_start

            if size > self.max_paragraph_len:
                sentences = [[s_start,s_start+size//2], [s_start+size//2,s_end]] + sentences
                continue
            
            is_valid = True
            
            for m_start, m_end in mentions:
                start_inside = m_start >= s_start and m_start < s_end
                end_inside = m_end > s_start and m_end <= s_end
                if start_inside and not end_inside:
                    _, ns_end = sentences.pop(0)

                    if m_start - s_start == 0 or ns_end - m_start == 0:
                        sentences = [[s_start, ns_end]] + sentences
                    else:
                        sentences = [[s_start, m_start], [m_start, ns_end]] + sentences
                    is_valid = False
                    break
            
            if is_valid:
                valid_sentences.append([s_start, s_end])

        return valid_sentences

    def sentences_to_paragraphs(self, sentences : Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # TODO : Docstring & Test

        paragraphs = []

        # start paragraph with first sentence of section
        p_start, p_end = sentences[0]

        for s_start, s_end in sentences[1:]:
            # If adding sentence to the paragraph make it larger than `self.max_paragraph`
            # Append current it into paragraph_spans and start a new paragraph
            if s_end - p_start > self.max_paragraph_len:
                paragraphs.append((p_start, p_end))
                p_start = s_start
            p_end = s_end

        # append the last paragraph of the section
        paragraphs.append((p_start, p_end))
        
        # make sure pragraphs length equals to section length
        assert any([e - s > self.max_paragraph_len for s, e in paragraphs]) == False, f'One of the paragraph is longer than {self.max_paragraph_len}'

        return paragraphs

    def preprocess_doc(self, doc : Dict[str, Any]) -> Dict[str, Any]:

        # TODO : Docstring & Test

        doc_id = doc['doc_id']
        words = doc['words']

        # If there is no words doc is empty so return None        
        if len(words) <= 0:
            return None

        mentions = [mention for mention in doc['ner'] if mention[-1] in self.entities]
        # If there is no mentions they will be no relation too so return None
        if len(mentions) <= 0:
            return None

        sections = [section for section in doc['sections'] if section[1] - section[0] > 0]
        assert len(sections) >= 1, f'len(words) ({len(words)}) is > 0 so len(sections) ({len(sections)}) should be > 0'

        sentences = [sentence for sentence in doc['sentences'] if sentence[1] - sentence[0] > 0]
        assert len(sentences) >= 1, f'len(words) ({len(words)}) is > 0 so len(sections) ({len(sentences)}) should be > 0'

        # Remove empty values coref and out of scope coref entity
        # TODO : Test
        corefs = {k:v for k, v in doc['coref'].items() if len(v) > 0 and RelDataLoader.get_ent_type(v[0], doc) in self.entities}
        # Is there is no corefs they will be no relation too so return None
        if len(corefs) <= 0:
            return None

        coref_to_idx, idx_to_coref = RelDataLoader.get_map_lists(corefs)
        # Map coref clusters to entity types by ids
        # TODO : Test
        coref_idx_to_entity_idx = {coref_to_idx[k]:self.entity_to_idx[RelDataLoader.get_ent_type(v[0], doc)] for k, v in corefs.items()}

        # Map relation and coref ids and map entity and coref ids
        # TODO : Test
        relation_idx_to_cluster_idx = []

        entity_idx_to_cluster_idx = [[] for k in self.idx_to_entity] # TODO : Try to see if we have differents candidate relation if we do it oterwise
        for coref_idx in idx_to_coref.keys():
            entity_idx_to_cluster_idx[coref_idx_to_entity_idx[coref_idx]].append(coref_idx)

        for rel_idx, rel in enumerate(RelDataLoader.filter_relations(doc['n_ary_relations'], corefs)):
            relation_idx_to_cluster_idx.append([])
            for entity in self.entities:
                if rel[entity] in coref_to_idx:
                    relation_idx_to_cluster_idx[rel_idx].append(coref_to_idx[rel[entity]])

        # No relation in this doc for given entities
        if len(relation_idx_to_cluster_idx) <= 0:
            return None

        cluster_to_relations_idx = defaultdict(set)
        for r, clist in enumerate(relation_idx_to_cluster_idx):
            for c in clist:
                cluster_to_relations_idx[c].add(r)

        candidate_relations = []
        candidate_relations_labels = []
        candidate_relations_types = []

        # for e in chain(combinations(used_entities, self.relation_cardinality), [(ent, ent) for ent in used_entities]):
        for e in itertools.combinations(list(self.idx_to_entity.keys()), 2):
            type_lists = [entity_idx_to_cluster_idx[x] for x in e]
            for clist in itertools.product(*type_lists):
                candidate_relations.append(clist)   
                common_relations = set.intersection(*[cluster_to_relations_idx[c] for c in clist])
                candidate_relations_labels.append(1 if len(common_relations) > 0 else 0)
                candidate_relations_types.append(tuple(e))

        # No valide candidate relation in this doc
        if len(candidate_relations) <= 0:
            return None

        # Extract section features
        tokenized_words = []
        section_items = []
        for i, section in enumerate(sections):
            _words, _sentences, _corefs = self.tokenize_section(section, sentences, corefs, words)
            _sentences, _paragraphs = self.chunk_section(_corefs, _sentences)
            section_items.append({
                'words' : _words,
                'sentences' : _sentences,
                'corefs' : _corefs,
                'paragraphs' : _paragraphs,
                'features' : [self.sfeature_to_idx[sf] for sf in RelDataLoader.get_section_features(section, i, doc)]
            })
            tokenized_words.extend(_words)
        # # ATM remove paragraph without any mention
        # # paragraph_spans = [p for p in paragraph_spans if span_has_ref(p)]
        # # paragraph_sf = [sf for p, sf in zip(paragraph_spans, paragraph_sf) if span_has_ref(p)]

        paragraphs = []
        paragraph_words = []
        paragraph_coref_spans = []
        paragraph_coref_types = []
        paragraph_coref_sfeatures = []
        paragraph_coref_ids = []

        for items in section_items:         
            for paragraph in items['paragraphs']:
                paragraphs.append(paragraph)
                paragraph_words.append(tokenized_words[paragraph[0]:paragraph[1]])
                paragraph_coref_spans.append([])
                paragraph_coref_types.append([])
                paragraph_coref_sfeatures.append([])
                paragraph_coref_ids.append([])
                for k, listv in items['corefs'].items():
                    ent_type = self.entity_to_idx[RelDataLoader.get_ent_type(listv[0], doc)]
                    for v in listv:
                        if is_span_inside(v, paragraph):
                            paragraph_coref_spans[-1].append(v)
                            paragraph_coref_types[-1].append(ent_type)
                            paragraph_coref_sfeatures[-1].append(items['features'])
                            paragraph_coref_ids[-1].append(coref_to_idx[k])
                # if len(paragraph_coref_spans) 


        def is_paragraph_relation(rels, coref_ids):
            for idx in coref_ids:
                for rel in rels:
                    if idx == rel[0] or idx == rel[1]:
                        return True
            return False

        outputs = []

        for i in range(0, len(candidate_relations), self.max_relation_len):

            rels_start = i
            rels_end = min(len(candidate_relations), rels_start+self.max_relation_len)

            _candidate_relations = candidate_relations[rels_start:rels_end]
            _candidate_relations_labels = candidate_relations_labels[rels_start:rels_end]
            _candidate_relations_types = candidate_relations_types[rels_start:rels_end]

            output = {
                'doc_id':doc_id,
                'words':words,

                'idx_to_coref':idx_to_coref,
                'coref_idx_to_entity_idx':coref_idx_to_entity_idx,
                'entity_idx_to_cluster_idx':entity_idx_to_cluster_idx,
                'relation_idx_to_cluster_idx':relation_idx_to_cluster_idx,

                'candidate_relations':_candidate_relations,
                'candidate_relations_labels':_candidate_relations_labels,
                'candidate_relations_types':_candidate_relations_types,
                
                'paragraphs':[],
                'paragraph_words':[],
                'paragraph_coref_spans':[],
                'paragraph_coref_types':[],
                'paragraph_coref_sfeatures':[],
                'paragraph_coref_ids':[]
            }

            for p, p_words, p_coref_spans, p_coref_types, p_coref_sfeatures, p_coref_ids in zip(
                paragraphs,
                paragraph_words,
                paragraph_coref_spans,
                paragraph_coref_types,
                paragraph_coref_sfeatures,
                paragraph_coref_ids,
            ):
                if is_paragraph_relation(_candidate_relations, p_coref_ids):
                    output['paragraphs'].append(p),
                    output['paragraph_words'].append(p_words),
                    output['paragraph_coref_spans'].append(p_coref_spans),
                    output['paragraph_coref_types'].append(p_coref_types),
                    output['paragraph_coref_sfeatures'].append(p_coref_sfeatures),
                    output['paragraph_coref_ids'].append(p_coref_ids)

            outputs.append(output)

        return outputs

    def collate_fn(self, batch_as_list):
        b = batch_as_list[0]
 
        idx_to_coref, coref_idx_to_entity_idx = (b['idx_to_coref'], b['coref_idx_to_entity_idx'])
        relation_idx_to_cluster_idx = b['relation_idx_to_cluster_idx']

        # Tensorize tokens
        tokens = [torch.LongTensor(words) for words in b['paragraph_words']]

        # Pad tokens
        input_ids = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Mask padded tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        doc_length = len(b['words'])
        
        relatif_spans = []
        spans_position = []
        spans_label_onehot = []
        spans_sf_onehot = []
        spans_ccluster_onehot = []

        for paragraph, coref_spans, coref_types, coref_sfeatures, coref_ids in zip(
            b['paragraphs'], 
            b['paragraph_coref_spans'], 
            b['paragraph_coref_types'], 
            b['paragraph_coref_sfeatures'],
            b['paragraph_coref_ids']):

            paragraph_start, _ = paragraph
            
            relatif_spans.append([])
            spans_position.append([])
            spans_label_onehot.append([])
            spans_sf_onehot.append([])
            spans_ccluster_onehot.append([])

            for coref_span, coref_span_type, coref_sf, coref_cluster in zip(coref_spans, coref_types, coref_sfeatures, coref_ids):
                coref_start, coref_end = coref_span
                relatif_spans[-1].append([coref_start-paragraph_start, coref_end-paragraph_start-1])
                spans_position[-1].append(np.mean([coref_start, coref_end]) / doc_length)
                spans_label_onehot[-1].append(RelDataLoader.get_onehot([coref_span_type], self.idx_to_entity))
                spans_sf_onehot[-1].append(RelDataLoader.get_onehot(coref_sf, self.idx_to_sfeature))
                spans_ccluster_onehot[-1].append(RelDataLoader.get_onehot([coref_cluster], idx_to_coref))

        # Pad relatif_spans & spans_position & spans_label_onehot & spans_sf_onehot
        max_span_count = max([len(s) for s in relatif_spans])

        for spans, positions, label_onehot, sf_onehot, ccluster_onehot in zip(relatif_spans, spans_position, spans_label_onehot, spans_sf_onehot, spans_ccluster_onehot):
            for i in range(len(spans), max_span_count):
                spans.append([0,0])
                positions.append(0)
                label_onehot.append(RelDataLoader.get_onehot([], self.idx_to_entity))
                sf_onehot.append(RelDataLoader.get_onehot([], self.idx_to_sfeature))
                ccluster_onehot.append(RelDataLoader.get_onehot([], idx_to_coref))

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
        
        return {
            'doc_id' : b['doc_id'],

            'cluster_to_type_arr': [t for _, t in sorted(coref_idx_to_entity_idx.items())],
            'entity_idx_to_cluster_idx':b['entity_idx_to_cluster_idx'],
            'relation_idx_to_cluster_idx':relation_idx_to_cluster_idx,

            'input_ids' : input_ids,
            'attention_mask' : attention_mask,

            'candidate_relations':b['candidate_relations'],
            'candidate_relations_labels':b['candidate_relations_labels'],
            'candidate_relations_types':b['candidate_relations_types'],

            'coref_spans':relatif_spans,
            'coref_spans_mask':spans_mask,
            'coref_spans_position': spans_position, # (B, P, 1)
            'coref_spans_labels': spans_label_onehot, # (B, P, L)
            'coref_spans_sf':spans_sf_onehot, # (B, S, SF)
            'coref_spans_cluster':spans_ccluster_onehot, # (B, S, C)
        }

    def create_dataloader(self, docs, is_train=False, batch_size=1, num_workers=1, **kwgs):
        docs = [self.preprocess_doc(doc) for doc in docs]
        docs = [doc for doc in docs if doc]
        docs = functools.reduce(lambda a, b : a + b, docs)
        return super().create_dataloader(docs, batch_size, num_workers, **kwgs)
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from helpers import read_jsonl

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    docs = read_jsonl('data/train.jsonl') + read_jsonl('data/dev.jsonl') + read_jsonl('data/test.jsonl')

    # for doc in docs:
    #     print(len(doc['n_ary_relations']))
    # Relation

    # loader = RelDataLoader(tokenizer)
    # docs = [loader.preprocess_doc(doc) for doc in tqdm(docs)]

    # print(len(docs))
    # docs = [doc for doc in docs if doc]
    # print(len(docs))
        
    # print(doc['doc_id'])
    # print(len(doc['sentences']), len())

    # doc = [doc for doc in docs if doc['doc_id'] == '0cfdcf2a0e345cdf7e680c30d136fdedb0eccb28'][0]
    # print(len(doc['sentences']), len(loader.chunk_sentences(doc['sentences'], doc['ner'])))

    # loader = RelDataLoader(tokenizer).create_dataloader(docs, batch_size=1, prefetch_factor=1)

    # Coref
    loader = CorefDataLoader(tokenizer).create_dataloader(docs, is_train=True, prefetch_factor=1)

    # NER 
    # loader = NERDataLoader(tokenizer).create_dataloader(docs, prefetch_factor=1)

    for b in loader:
        print(b)
        break