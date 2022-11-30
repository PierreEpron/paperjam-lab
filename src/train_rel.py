from pathlib import Path

from tqdm import tqdm
from helpers import read_jsonl, write_jsonl
from training import create_config, train_model
from model import BertRel

if __name__ == '__main__': 
    name = "rel_alpha"
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
        model_name='allenai/scibert_scivocab_uncased', train_batch_size=1, val_batch_size=1).rel
    
    model = BertRel(model_name=config.model_name)

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
            'shapes':[outputs['shape'] for _ in outputs['preds']]
        })


    write_jsonl(dirpath / 'test_results.jsonl', test_results)
