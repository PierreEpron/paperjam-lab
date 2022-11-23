from collections import defaultdict
from itertools import chain, combinations, product
from typing import Dict, List
from tqdm import tqdm
import sys, traceback

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel

from seqeval.metrics.sequence_labeling import get_entities

from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder, PytorchTransformer
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import util
from allennlp.modules import FeedForward, TimeDistributed

from helpers import select_first_subword
from dataset import NERDataLoader, CorefDataLoader, RelDataLoader

from seqeval.metrics import f1_score as ner_f1
from sklearn.metrics import f1_score as binary_f1

class BertRel(nn.Module):
    def __init__(self,
        model_name='allenai/scibert_scivocab_uncased',
        lexical_dropout = .2,
        context_hidden_size=200,
        relation_cardinality=2):

        super().__init__()
        
        self.relation_cardinality = relation_cardinality

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_layer = AutoModel.from_pretrained(model_name)

        self.lexical_dropout = nn.Dropout(p=lexical_dropout)

        self.context_layer = LstmSeq2SeqEncoder(768, context_hidden_size, bidirectional=True)
        
        self.attended_extractor = SelfAttentiveSpanExtractor(self.context_layer.get_output_dim())
        
        self.endpoint_extractor = EndpointSpanExtractor(self.context_layer.get_output_dim(), combination="x,y")

        self.bias_vectors = torch.nn.Parameter(torch.zeros((1, 4, 1210)))

        feedforward = nn.Sequential(
            nn.Sequential(
                nn.Linear(1210*self.relation_cardinality, 150),
                nn.GELU(),
                nn.Dropout(p=.2),
            ),
            nn.Sequential(
                nn.Linear(150, 150),
                nn.GELU(),
                nn.Dropout(p=.2),
            )
        ) 
        self.antecedent_feedforward = TimeDistributed(feedforward)
        
        self.antecedent_scorer = TimeDistributed(torch.nn.Linear(150, 1))

        self.data_processor = RelDataLoader(self.tokenizer)

    def forward(self, x, compute_loss=False):

        torch.cuda.empty_cache()

        input_ids, attention_mask = (x['input_ids'], x['attention_mask'])
        # print('input_ids.shape : ', input_ids.shape)
        # print('attention_mask.shape : ', attention_mask.shape)
        
        spans, spans_mask = (x['coref_spans'], x['coref_spans_mask'])
        # print('spans.shape : ', spans.shape)
        # print('spans : ', spans)
        # print('spans_mask.shape : ', spans_mask.shape)

        span_position, span_type_labels_one_hot, span_section_features, span_clusters = (x['coref_spans_position'], x['coref_spans_labels'], x['coref_spans_sf'], x['coref_spans_cluster'])
        # print('span_position.shape : ', span_position.shape)
        # print('span_type_labels_one_hot.shape : ', span_type_labels_one_hot.shape)
        # print('span_section_features.shape : ', span_section_features.shape)
        # print('span_clusters.shape : ', span_clusters.shape)

        if span_clusters.sum() == 0:
            # print(f'SPAN CLUSTERS IS EMPTY FOR DOC : {x["doc_id"]}')
            return {'doc_id':x['doc_id'], "loss": None}

        cluster_to_type_arr = x['cluster_to_type_arr']
        # entity_idx_to_cluster_idx = x['entity_idx_to_cluster_idx']
        # relation_idx_to_cluster_idx = x['relation_idx_to_cluster_idx']

        candidate_relations, candidate_relations_labels, candidate_relations_types = (x['candidate_relations'], x['candidate_relations_labels'], x['candidate_relations_types'])


        text_embeddings = self.bert_layer(input_ids, attention_mask).last_hidden_state
        # print('text_embeddings.shape : ', text_embeddings.shape)

        # TODO : flatten like scirex here ?

        contextualized_embeddings = self.context_layer(text_embeddings, attention_mask)
        # print('contextualized_embeddings.shape : ', contextualized_embeddings.shape)

        attented_span_embeddings = self.attended_extractor(contextualized_embeddings, spans, attention_mask, spans_mask)
        # print('attented_span_embeddings.shape : ', attented_span_embeddings.shape)

        # TODO : I don't know why they do that :'(
        # spans_relu =  nn.functional.relu(spans.float()).long()
        # # print('spans_relu.shape : ', spans_relu.shape)
        # # print(torch.equal(spans, spans_relu)) # Equal True ...

        endpoint_span_embeddings = self.endpoint_extractor(contextualized_embeddings, spans, attention_mask, spans_mask)
        # print('endpoint_span_embeddings.shape : ', endpoint_span_embeddings.shape)
        
        span_embeddings = torch.cat([endpoint_span_embeddings, attented_span_embeddings], -1)
        # print('span_embeddings.shape : ', span_embeddings.shape)

        span_features = torch.cat([span_position, span_type_labels_one_hot.float(), span_section_features.float()], dim=-1) 
        # print('span_features.shape : ', span_features.shape)

        featured_span_embeddings = torch.cat([span_embeddings, span_features], dim=-1)
        # print('featured_span_embeddings.shape : ', featured_span_embeddings.shape)

        sum_embeddings = (featured_span_embeddings.unsqueeze(2) * span_clusters.unsqueeze(-1)).sum(1)
        # print('sum_embeddings.shape : ', sum_embeddings.shape)

        length_embeddings =  (span_clusters.unsqueeze(-1).sum(1) + 1e-5)
        # print('length_embeddings.shape : ', length_embeddings.shape)

        cluster_span_embeddings = sum_embeddings / length_embeddings
        # print('cluster_span_embeddings.shape : ', cluster_span_embeddings.shape)

        paragraph_cluster_mask = (span_clusters.sum(1) > 0).float().unsqueeze(-1)
        # print('paragraph_cluster_mask.shape : ', paragraph_cluster_mask.shape)

        cluster_type_embeddings = self.bias_vectors[:, cluster_to_type_arr]
        # print('cluster_type_embeddings.shape : ', cluster_type_embeddings.shape)

        paragraph_cluster_embeddings = cluster_span_embeddings * paragraph_cluster_mask + cluster_type_embeddings * (1 - paragraph_cluster_mask)
        # print('paragraph_cluster_embeddings.shape : ', paragraph_cluster_embeddings.shape)

        paragraph_cluster_embeddings = torch.cat(
            [paragraph_cluster_embeddings, self.bias_vectors.expand(paragraph_cluster_embeddings.shape[0], -1, -1)],
            dim=1,
        ) 
        # print('paragraph_cluster_embeddings.shape : ', paragraph_cluster_embeddings.shape)  # (P, C+T, E)

        # used_entities = list(self.data_processor.idx_to_entity.keys())
        # bias_vectors_clusters = {x: i + n_true_clusters for i, x in enumerate(used_entities)}

        # cluster_to_relations_id = defaultdict(set)
        # for r, clist in enumerate(relation_idx_to_cluster_idx):
        #     # for t in bias_vectors_clusters.values():
        #     #     cluster_to_relations_id[t].add(r)
        #     for c in clist:
        #         cluster_to_relations_id[c].add(r)

        # candidate_relations = []
        # candidate_relations_labels = []
        # candidate_relations_types = []

        # for e in chain(combinations(used_entities, self.relation_cardinality), [(ent, ent) for ent in used_entities]):
        # for e in combinations(used_entities, self.relation_cardinality):
        #     type_lists = [entity_idx_to_cluster_idx[x] for x in e]
        #     for clist in product(*type_lists):
        #         candidate_relations.append(clist)   
        #         common_relations = set.intersection(*[cluster_to_relations_id[c] for c in clist])
        #         candidate_relations_labels.append(1 if len(common_relations) > 0 else 0)
            #     candidate_relations_types.append(self._relation_type_map[tuple(e)])

        # if len(candidate_relations) == 0:
        #     # print(f'CANDIDATE RELATION IS EMPTY FOR DOC : {x["doc_id"]}')
        #     return {'doc_id':x['doc_id'], "loss": None}

        candidate_relations_tensor = torch.LongTensor(candidate_relations).to(text_embeddings.device)
        # print('candidate_relations_tensor.shape : ', candidate_relations_tensor.shape)

        candidate_relations_labels_tensor = torch.LongTensor(candidate_relations_labels).to(text_embeddings.device)

        relation_embeddings = util.batched_index_select(
            paragraph_cluster_embeddings,
            candidate_relations_tensor.unsqueeze(0).expand(paragraph_cluster_embeddings.shape[0], -1, -1),
        )

        relation_embeddings = relation_embeddings.view(relation_embeddings.shape[0], relation_embeddings.shape[1], -1)
        # print('relation_embeddings.shape : ', relation_embeddings.shape)

        relation_embeddings = self.antecedent_feedforward(relation_embeddings)
        # print('relation_embeddings.shape : ', relation_embeddings.shape)

        relation_embeddings = relation_embeddings.max(0, keepdim=True)[0]
        # print('relation_embeddings.shape : ', relation_embeddings.shape)

        relation_logits = self.antecedent_scorer(relation_embeddings).squeeze(-1).squeeze(0)
        # print('relation_logits.shape : ', relation_logits.shape)

        outputs={'doc_id':x['doc_id'], 'logits':relation_logits, 'golds':candidate_relations_labels}

        if compute_loss:
            outputs["loss"] = nn.functional.binary_cross_entropy_with_logits(
                relation_logits,
                candidate_relations_labels_tensor.float(),
                reduction="mean",
            )

        outputs["probs"] = torch.sigmoid(outputs['logits'])

        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            outputs['preds'] = (outputs["probs"] > .5).int().cpu().numpy() if 'probs' in outputs else []
        return outputs

    def metric(self, preds, golds, **kwargs):
        return binary_f1(preds, golds, **kwargs)


class BertCoref(nn.Module):
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_layer = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(p=.0)

        self.classification_layer = nn.Sequential(
            nn.Sequential(
                nn.Linear(768, 200),
                nn.ReLU(),
                nn.Dropout(p=.2),
            ),
            nn.Sequential(
                nn.Linear(200, 2),
                nn.Dropout(p=0),
            )
        )       

        self.loss = nn.CrossEntropyLoss()
        
        self.data_processor = CorefDataLoader(self.tokenizer)

    def forward(self, x, compute_loss=False):
        
        pooled = self.dropout(self.bert_layer(x['tokens'], x['attention_mask']).pooler_output)
        logits = self.classification_layer(pooled)

        outputs = {"logits": logits, 'golds':x['golds']}

        if compute_loss:
            outputs["loss"] = self.loss(logits, torch.LongTensor(x['golds']).to(logits.device).long().view(-1))

        outputs["probs"] = nn.functional.softmax(outputs['logits'], dim=-1)[..., 1]

        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            outputs['preds'] = (outputs['probs'] > .5).int().cpu().numpy()
        return outputs

    def metric(self, preds, golds, **kwargs):
        return binary_f1(preds, golds, **kwargs)

class BertWordCRF(nn.Module):
    def __init__(
        self, tag_to_id, model_name='allenai/scibert_scivocab_uncased', 
        tag_format='BIO', word_encoder='transformer', mode='word'):

        super().__init__()

        self.mode = mode

        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}

        constraints = allowed_transitions(tag_format, self.id_to_tag)

        n_labels = len(self.id_to_tag)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_layer = AutoModel.from_pretrained(model_name)

        self.h_size = self.bert_layer.config.hidden_size

        if word_encoder == 'lstm':
            self.word_encoder = LstmSeq2SeqEncoder(
                self.h_size, self.h_size // 2, bidirectional=True, dropout=0.1)
        elif word_encoder == 'transformer':
            self.word_encoder = PytorchTransformer(
                self.h_size, num_layers=1, positional_encoding="sinusoidal")
        else:
            raise ValueError

        self.output_layer = nn.Linear(self.h_size, n_labels)

        self.crf_layer = ConditionalRandomField(
            n_labels, constraints=constraints)

        self.data_processor = NERDataLoader(self.tokenizer)

    def forward(self, x, compute_loss=False):

        h = self.bert_layer(
            x['input_ids'], x['attention_mask']).last_hidden_state

        if self.mode == 'word':

            out_sel = select_first_subword(
                h, x['subword_mask'], x['seq_length'])

            h, word_mask = out_sel['first_subword'], out_sel['word_mask']

            h = self.word_encoder(h, word_mask)

        if self.mode == 'subword':

            h = self.word_encoder(h, x['attention_mask'].ne(0))

            out_sel = select_first_subword(
                h, x['subword_mask'], x['seq_length'])

            h, word_mask = out_sel['first_subword'], out_sel['word_mask']

        logits = self.output_layer(h)

        outputs = {'logits': logits, 'golds':x['golds'], 'word_mask':word_mask}

        if compute_loss:
            loss = - self.crf_layer(logits, x['word_label'], word_mask)
            outputs['loss'] = loss

        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            prediction = self.crf_layer.viterbi_tags(outputs['logits'], outputs['word_mask'])
            outputs['preds'] = [self.id_to_IOB(i[0]) for i in prediction]
        return outputs

    def id_to_IOB(self, sequence):
        out = []
        for i in sequence:
            out.append(self.id_to_tag[i])
        return out

    def predict_from_tokens(self, tokens: List[List[str]], **loader_kwgs):

        data = [{'tokens': t} for t in tokens]

        loader = self.data_processor.create_dataloader(data, batch_size=1, **loader_kwgs)

        device = next(self.parameters()).device

        all_preds = []

        for x in tqdm(loader):

            for k, v in x.items():
                if torch.is_tensor(v):
                    x[k] = v.to(device)
            pred = self.predict(x)
            all_preds.extend(pred)

        return all_preds

    def extract_entities(self, list_tokenised_sequence: List[List[str]], **loader_kwgs):

        predictions = self.predict_from_tokens(
            list_tokenised_sequence, **loader_kwgs)

        k = []
        for pred, d in zip(predictions, list_tokenised_sequence):
            s = []
            pred = get_entities(pred)
            for p in pred:
                ent_type, start, end = p
                entity = ' '.join(d[start: end + 1])
                out = {'entity': entity, 'type': ent_type,
                       'span': [start, end]}
                s.append(out)
            k.append(s)

        return k

    def metric(self, preds, golds, **kwargs):
        return ner_f1(preds, golds, **kwargs)

if __name__ == "__main__":
    from helpers import read_jsonl

    data = read_jsonl('data/train.jsonl')

    model = BertRel()

    # data = [doc for doc in data if doc['doc_id'] == '3b9732bb07dc99bde5e1f9f75251c6ea5039373e']

    loader = model.data_processor.create_dataloader(data, batch_size=1, prefetch_factor=1)

    import sys

    for b in tqdm(loader):
        try:
            model.forward(b)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)