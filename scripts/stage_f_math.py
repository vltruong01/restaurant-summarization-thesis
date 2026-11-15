"""Frozen deterministic Stage F reduction, broader grouping and diversity functions."""
import numpy as np


def quality_order(rows):
    return sorted(range(len(rows)), key=lambda i:(-round(rows[i]['clip_score'],6),-round(rows[i]['relevance_margin'],6),rows[i]['photo_id']))


def redundancy_reduce(rows, similarity, threshold):
    kept=[];mapping={}
    matrix=np.round(similarity.astype(np.float64),6)
    for i in quality_order(rows):
        duplicate=[j for j in kept if matrix[i,j]>=threshold]
        if not duplicate:kept.append(i)
        else:mapping[i]=min(duplicate,key=lambda j:(-matrix[i,j],rows[j]['photo_id']))
    return kept,mapping


def image_duplicate_reduce(rows,pixel_hashes,phash_values,threshold):
    kept=[];removed={}
    for i in quality_order(rows):
        duplicate=[j for j in kept if pixel_hashes[i]==pixel_hashes[j] or (phash_values[i]^phash_values[j]).bit_count()<=threshold]
        if not duplicate:kept.append(i)
        else:removed[i]=min(duplicate,key=lambda j:((phash_values[i]^phash_values[j]).bit_count(),rows[j]['photo_id']))
    return kept,removed


def complete_link_groups(rows, similarity, threshold):
    matrix=np.round(similarity.astype(np.float64),6)
    clusters=[(i,) for i in sorted(range(len(rows)),key=lambda i:rows[i]['photo_id'])]
    while True:
        eligible=[]
        for a in range(len(clusters)):
            for b in range(a+1,len(clusters)):
                score=min(matrix[i,j] for i in clusters[a] for j in clusters[b])
                if score>=threshold:
                    ids=tuple(sorted(rows[i]['photo_id'] for i in clusters[a]+clusters[b]))
                    eligible.append((-score,ids,a,b))
        if not eligible:break
        _,_,a,b=min(eligible);merged=tuple(sorted(clusters[a]+clusters[b],key=lambda i:rows[i]['photo_id']))
        clusters=[c for k,c in enumerate(clusters) if k not in [a,b]]+[merged]
        clusters.sort(key=lambda c:tuple(rows[i]['photo_id'] for i in c))
    return {i:'GROUP_'+str(g) for g,c in enumerate(clusters) for i in c}


def farthest_first(rows, similarity, candidates):
    quality=quality_order(rows);rank={i:n for n,i in enumerate(quality)}
    remaining=set(candidates)
    if not remaining:return []
    selected=[min(remaining,key=lambda i:rank[i])];remaining.remove(selected[0])
    matrix=np.round(similarity.astype(np.float64),6)
    while remaining:
        i=min(remaining,key=lambda i:(-min(1-matrix[i,j] for j in selected),rank[i]))
        selected.append(i);remaining.remove(i)
    return selected


def threshold_search(scores,labels,grid,passes_above=True,minimum_class=10):
    positive=sum(labels);negative=len(labels)-positive
    if min(positive,negative)<minimum_class:
        raise ValueError('Insufficient actual human positive/negative labels; do not invent a threshold.')
    result=[]
    scores=[round(float(s),6) for s in scores]
    for t in grid:
        predictions=[s>=t if passes_above else s<=t for s in scores]
        tp=sum(p and y for p,y in zip(predictions,labels));fp=sum(p and not y for p,y in zip(predictions,labels));tn=sum(not p and not y for p,y in zip(predictions,labels))
        recall=tp/positive;precision=tp/(tp+fp) if tp+fp else 0
        balanced=(recall+tn/negative)/2
        result.append({'threshold':float(t),'balanced_accuracy':balanced,'positive_precision':precision,'positive_recall':recall,'positive_labels':positive,'negative_labels':negative})
    return max(result,key=lambda r:(r['balanced_accuracy'],r['positive_precision'],r['positive_recall'],r['threshold'])),result
