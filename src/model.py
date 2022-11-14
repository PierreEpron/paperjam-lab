from typing import Dict, List
from tqdm import tqdm

from torch import nn
import torch

from transformers import AutoTokenizer, AutoModel

from seqeval.metrics.sequence_labeling import get_entities

from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder, PytorchTransformer

from helpers import select_first_subword
from dataset import NERDataLoader, CorefDataLoader

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

        outputs = {"logits": logits, "metadata":x["metadata"]}

        if compute_loss:
            outputs["loss"] = self.loss(logits, x['true'].long().view(-1))

        return outputs

    def predict_probs(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            probs = nn.functional.softmax(outputs['logits'], dim=-1)[..., 1]
            outputs["probs"] = probs
        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            probs = nn.functional.softmax(outputs['logits'], dim=-1)[..., 1]
            prediction = (probs > .5).int()
        return prediction

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

        outputs = {'logits': logits, 'word_mask': out_sel['word_mask']}

        if compute_loss:
            loss = - self.crf_layer(logits, x['word_label'], word_mask)
            outputs['loss'] = loss

        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x, compute_loss=False)
            prediction = self.crf_layer.viterbi_tags(
                outputs['logits'], outputs['word_mask'])
        return [self.id_to_IOB(i[0]) for i in prediction]

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

if __name__ == "__main__":
    from helpers import read_jsonl

    data = read_jsonl('data/train.jsonl')

    model = BertCoref()

    loader = model.data_processor.create_dataloader(data, batch_size=8)

    for b in loader:
        print(model.predict(b))
        break