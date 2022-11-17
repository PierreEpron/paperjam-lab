from functools import reduce
from helpers import read_jsonl
from pathlib import Path
import json

def span_match(span_1, span_2):
    sa, ea = span_1
    sb, eb = span_2
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou

def overlap_score(cluster_1, cluster_2):
    matched = 0
    for s1 in cluster_1.values():
        print(cluster_2)
        matched += 1 if any([span_match(s1, s2) > 0.5 for s2 in cluster_2.values()]) else 0

    return matched / len(cluster_1)

def compute_metrics(predicted_clusters, gold_clusters):
    matched_predicted = []
    matched_gold = []
    for i, p in enumerate(predicted_clusters):
        for j, g in enumerate(gold_clusters):
            if overlap_score(p, g) > 0.5:
                matched_predicted.append(i)
                matched_gold.append(j)

    matched_predicted = set(matched_predicted)
    matched_gold = set(matched_gold)

    metrics = {
        "p": len(matched_predicted) / (len(predicted_clusters) + 1e-7),
        "r": len(matched_gold) / (len(gold_clusters) + 1e-7),
    }
    metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

    return metrics

if __name__ == "__main__":

    model_path = "output/coref/coref.ckpt"
    test_path = "data/test.jsonl"
    pred_path = Path(model_path).with_name('coref_clusters.jsonl')

    test_clusters = [doc['coref'] for doc in read_jsonl(test_path)]
    pred_clusters = [doc['clusters'] for doc in read_jsonl(pred_path)]
    
    Path('a.json').write_text(json.dumps(test_clusters[0]), encoding='utf-8')
    Path('b.json').write_text(json.dumps(pred_clusters[0]), encoding='utf-8')


    print(compute_metrics(pred_clusters, test_clusters))