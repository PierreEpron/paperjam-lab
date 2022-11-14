from tqdm import tqdm

import numpy as np

from inference import load_model
from helpers import read_jsonl, write_jsonl

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def generate_matrix_for_document(document, span_field, matrix_field) :
    span2idx = {tuple(k):i for i, k in enumerate(document[span_field])}
    matrix = np.zeros((len(span2idx), len(span2idx)))
    for e1, e2, score in document[matrix_field] :
        matrix[span2idx[tuple(e1)], span2idx[tuple(e2)]] = score
        
    return matrix

def cluster_with_clustering(matrix) :
    scores = []
    matrix = (matrix + matrix.T) + np.eye(*matrix.shape)
    for n in range(2, matrix.shape[0] if matrix.shape[0] > 2 else 3) :
        clustering = AgglomerativeClustering(n_clusters=n, linkage='complete', affinity='precomputed').fit(1 - matrix)
        if matrix.shape[0] > 2 :
            scores.append(silhouette_score(1 - matrix, clustering.labels_, metric='precomputed'))
        else :
            scores.append(1)

    best_score = max(scores)
    best_n = scores.index(best_score) + 2
    clustering = AgglomerativeClustering(n_clusters=best_n, linkage='complete', affinity='precomputed').fit(1 - matrix)
    return clustering.n_clusters_, clustering.labels_

def map_back_to_spans(document, span_field, labels) :
    idx2span = {i:tuple(k) for i, k in enumerate(document[span_field])}
    span_to_label_map = {}
    for i, span in idx2span.items() :
        span_to_label_map[span] = labels[i]
    return span_to_label_map

def clusterize(file_path, **loader_kwgs):
    model = load_model(model_path)
    data = read_jsonl(file_path)
    
    cluster_outputs = []

    for item in data[:1]:
        doc = {'doc_id':item['doc_id']}
        loader = model.data_processor.create_dataloader([item], **loader_kwgs)
        scores = []
        metadata = []
        for x in tqdm(loader):
            output = model.predict_probs(x)
            scores.extend(output['probs'].cpu().numpy())
            metadata.extend(output['metadata'])
            break
        doc['pairwise_coreference_scores'] = []
        doc['spans'] = []
        for s, m in zip(scores, metadata): 
            doc['pairwise_coreference_scores'].append((m['span_1'], m['span_2'], s))
            doc['spans'].extend([m['span_1'], m['span_2']])
        doc['spans'] = sorted(list(set(doc['spans'])))

    matrix = generate_matrix_for_document(doc, 'spans', 'pairwise_coreference_scores')
    n_clusters, cluster_labels = cluster_with_clustering(matrix)
    span_to_cluster_label = map_back_to_spans(doc, 'spans', cluster_labels)

    clusters = [{'spans' : [], 'words': set(), 'types' : set()} for _ in range(n_clusters)]
    for s, l in span_to_cluster_label.items() :
        clusters[l]['spans'].append(s)

    coref_clusters = {str(i): v["spans"] for i, v in enumerate(clusters)}

    cluster_outputs.append({'doc_id' : doc['doc_id'], 'spans' : doc['spans'], 'clusters' : coref_clusters})

    return cluster_outputs
#         'pairwise_coreference_scores' : List[(s_1, e_1), (s_2, e_2), float (3 sig. digits) in [0, 1]]

        # 'spans' : List[Tuple[int, int]]

if __name__ == '__main__': 

    model_path = "output/coref/coref.ckpt"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"

    f1 = None

    write_jsonl('data/coref_clusters.jsonl', clusterize(test_path, batch_size=64, num_workers=8))