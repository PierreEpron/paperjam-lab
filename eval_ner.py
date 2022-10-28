from pathlib import Path
import os, sys

import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy import displacy
from thinc.api import Config

LABELS = ["OBJ_LOC"]
PATH = 'data/train/suv_demo/'
MODEL_PATH = PATH + 'output/model-best'
TRAIN_PATH = PATH + "corpus/train.spacy"
DEV_PATH = PATH + "corpus/dev.spacy"
TEST_PATH = PATH + "corpus/test.spacy"
HTML_PATH = PATH + 'output/'

def ner_html(path, examples, manual=False, options={}, filter_labels=[]):
    """
    Shorcut for save docs in html with displacy

    Parameters
    ----------
    path : str or Path, path of data to load
    examples : list, see doc of displacy.
    manual : bool, see doc of displacy.
    options : dict, see doc of displacy.
    """
    path = Path(path) if isinstance(path, str) else path
    docs = []
    for i, example in enumerate(examples):
        docs.append({
                      "text": example.reference.text,
                      "ents": [{"start":ent.start_char, "end":ent.end_char , "label":ent.label_ } for ent in example.reference.ents if ent.label_ in filter_labels],
                      "title": '{} TRUE'.format(i)
                    })
        docs.append({
                        "text": example.predicted.text,
                        "ents": [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in example.predicted.ents if ent.label_ in filter_labels],
                        "title": '{} PRED'.format(i)
                    })

    path.write_text(displacy.render(docs, style='ent', manual=manual, options=options, page=True), encoding='utf-8')
 
def parse_pred(i, doc):
    '''
    Parse prediction for displacy html view
    Parameters
    ----------
    i : int , document id
    doc : spacy.Doc, doc to parse.
    
    Returns
    -------
    result : dict, formatted doc for displacy.
    '''
    ents = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_ } for ent in doc.ents]
    ents.sort(key=lambda x: x['start'])
    return {"text": doc.text,
            "ents": ents,
            "title": f'{i} PRED'}
            
def parse_true(i, item):
    '''
    Parse gold for displacy html view
    Parameters
    ----------
    i : int , document id
    doc : spacy.Doc, doc to parse.
    
    Returns
    -------
    result : dict, formatted doc for displacy.
    '''
    ents = [{"start": ent[0], "end": ent[1], "label": ent[2] } for ent in item['labels']]
    ents.sort(key=lambda x: x['start'])
    return {"text": item['text'],
            "ents": ents,
            "title": f'{i} TRUE'}

def load_model(path):
    path = Path(path) if isinstance(path, str) else path
    config = Config().from_disk(path / "config.cfg")
    lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
    nlp = lang_cls.from_config(config)
    nlp.from_disk(path)
    return nlp

def load_examples(doc_bin, nlp):
    examples = []
    docs = list(doc_bin.get_docs(nlp.vocab))
    for i, doc in enumerate(docs):
        examples.append(Example(nlp(doc.text), doc))
        print(f"{i + 1}/{len(docs)} ({(i + 1) / len(docs) * 100})")
    return examples

nlp = load_model(MODEL_PATH)
scorer = Scorer(nlp)

train_examples = load_examples(DocBin().from_disk(TRAIN_PATH), nlp)
dev_examples = load_examples(DocBin().from_disk(DEV_PATH), nlp)
test_examples = load_examples(DocBin().from_disk(TEST_PATH), nlp)

scores = scorer.score(test_examples)
print(f"{scores['ents_p']}, {scores['ents_r']}", {scores['ents_f']})
print(f"{scores['ents_per_type']}")

ner_html(HTML_PATH+"/train_html.html", train_examples, manual=True, filter_labels=LABELS)
ner_html(HTML_PATH+"/dev_html.html", dev_examples, manual=True, filter_labels=LABELS)
ner_html(HTML_PATH+"/test_html.html", test_examples, manual=True, filter_labels=LABELS)
