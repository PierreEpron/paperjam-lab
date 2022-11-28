from pathlib import Path

from tqdm import tqdm
from coref_clusterize import clusterize
from helpers import read_jsonl, write_jsonl
from training import create_config, train_model
from model import BertCoref
from Levenshtein import ratio

if __name__ == '__main__': 
    name = "coref_alpha"
    dirpath= Path(f"outputs/{name}/")
    train_path = Path("data/train.jsonl")
    dev_path = Path("data/dev.jsonl")
    test_path = Path("data/test.jsonl")

    if not dirpath.is_dir():
        dirpath.mkdir()

    name = f"{name}_{len([path for path in dirpath.glob('*') if path.is_dir()])}"
    dirpath /= name

    dirpath.mkdir()

    config = create_config(
        name, dirpath=str(dirpath), train_path=str(train_path), dev_path=str(dev_path), max_epoch=10, num_workers=1, 
        model_name='allenai/scibert_scivocab_uncased', train_batch_size=64, val_batch_size=64).coref
    
    model = BertCoref(model_name=config.model_name)

    train_model(model, config)

    test_loader = model.data_processor.create_dataloader(
        read_jsonl(test_path), 
        batch_size=1, 
        num_workers=1, 
        shuffle=False
    )

    test_results = []
    docs = {}

    for x in tqdm(test_loader):

        outputs = model.predict(x)
        test_results.append({
            'doc_id':outputs['doc_id'],
            'golds':[int(gold) for gold in outputs['golds']],
            'preds':[int(pred) for pred in outputs['preds']],
            'span_1':[w for w in outputs['span_1']],
            'span_2':[w for w in outputs['span_2']],
            'type_1':[t for t in outputs['type_1']],
            'type_2':[t for t in outputs['type_2']],
            'levs':[ratio(' '.join(w1).lower(), ' '.join(w2).lower()) for w1, w2 in zip(outputs['word_1'], outputs['word_2'])]
        })

        for doc_id in outputs['doc_id']:
            if doc_id not in docs:
                docs[doc_id] = {'spans':[],'types':[],'pairwise_coreference_scores':[]}

            # set()
            docs[doc_id]['spans'].extend([(outputs['span_1'][0], outputs['type_1'][0]), (outputs['span_2'][0], outputs['type_2'][0])])
            docs[doc_id]['spans'] = sorted(list(set(docs[doc_id]['spans'])))
            docs[doc_id]['pairwise_coreference_scores'].append((outputs['span_1'][0], outputs['span_2'][0], outputs['probs'].item()))

    write_jsonl(dirpath / 'test_results.jsonl', test_results)
    write_jsonl(dirpath / 'clusters.jsonl', clusterize(docs))
