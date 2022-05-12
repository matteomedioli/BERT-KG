import asyncio
import math
import os
import aiohttp
import pandas as pd
from aiohttp import FormData
from nltk.corpus import wordnet as wn
import nltk
import numpy as np

# nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import torch
import json
import requests


def lcs(X, Y, m, n):
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]

    # Create a character array to store the lcs string
    lcs = [""] * (index + 1)
    lcs[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs


def wup_similarity_wordnet(synset_i, synset_j, score=None, simulate_root=True):
    need_root = synset_i._needs_root()
    subsumers = synset_i.lowest_common_hypernyms(
        synset_j, simulate_root=simulate_root and need_root, use_min_depth=True
    )
    # If no LCS was found return None
    if len(subsumers) == 0:
        return 0
    subsumer = synset_i if synset_i in subsumers else subsumers[0]
    depth = subsumer.max_depth() + 1
    len1 = synset_i.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    len2 = synset_j.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    if len1 is None or len2 is None:
        return 0
    len1 += depth
    len2 += depth
    if score:
        log_score = math.log(score + 1)
        return ((2.0 * depth) + log_score) / ((len1 + len2) + log_score)
    else:
        return (2.0 * depth) / (len1 + len2)


def Gloss(synset):
    gloss = str(synset.definition())
    pattern = re.compile('[\W_]+')
    gloss = pattern.sub(' ', gloss)
    gloss = gloss.split(" ")
    return gloss


def Related(synset):
    l = []
    e = synset.hyponyms()
    if e:
        l.append(e)

    e = synset.hypernyms()
    if e:
        l.append(e)

    e = synset.member_meronyms()
    if e:
        l.append(e)

    e = synset.substance_meronyms()
    if e:
        l.append(e)

    e = synset.part_meronyms()
    if e:
        l.append(e)

    e = synset.part_holonyms()
    if e:
        l.append(e)

    e = synset.substance_holonyms()
    if e:
        l.append(e)

    e = synset.member_holonyms()
    if e:
        l.append(e)

    e = synset.also_sees()
    if e:
        l.append(e)

    e = synset.similar_tos()
    if e:
        l.append(e)

    l = [item for sublist in l for item in sublist]

    return l


def Lemma(synset):
    words = []
    for lemma in synset.lemmas():
        pattern = re.compile('[\W_]+')
        l = pattern.sub(' ', lemma.name())
        l = l.split(" ")
        for w in l:
            words.append(w)
    return words


def descriptor(synset):
    desc = []
    desc += Lemma(synset) + Gloss(synset)
    for r in Related(synset):
        desc += Gloss(r)
    desc = [x for x in desc if x not in stopwords.words('english')]
    desc.sort()
    return np.unique(desc)


def score(synset_i, synset_j):
    desc_i = descriptor(synset_i)
    desc_j = descriptor(synset_j)
    desc_i = list(filter(None, desc_i))
    desc_j = list(filter(None, desc_j))

    k = 0
    N = 0
    len_i_init = len(desc_i)
    len_j_init = len(desc_j)
    lcs_list = lcs(desc_i, desc_j, len_i_init, len_j_init)
    N += len(lcs_list)
    k += 1
    while len(lcs_list) > 1:
        for word in lcs_list:
            if word in desc_i and word in desc_j:
                desc_i.remove(word)
                desc_j.remove(word)
                len_i = len(desc_i)
                len_j = len(desc_j)
                lcs_list = lcs(desc_i, desc_j, len_i, len_j)
                N += len(lcs_list) + len(lcs_list)
                k += 1
    return N / k


def max_cj(ci, cjs):
    for cj in cjs:
        return wup_similarity_wordnet(ci, cj, score(ci, cj))


def sentence_synsets(sentence, node_embeddings, device, config, evaluate_neighbors=True, neighbors_d=2):
    sw = stopwords.words("english")
    synsets = {}
    sentence_node_embeddings = []
    for i, target in enumerate(sentence):
        if "[CLS]" not in target and "[MASK]" not in target and "[PAD]" not in target and target.lower() not in sw:
            target_synsets = wn.synsets(target)
            target_synsets = [s for s in target_synsets if s.name() in node_embeddings.keys()]
            if evaluate_neighbors and target not in synsets.keys() and target_synsets is not None:
                neighbors = []
                for x in range(1, neighbors_d):
                    neg = (i - x) % len(sentence)
                    pos = (i + x) % len(sentence)
                    if neg < i:
                        neighbors.append(sentence[neg])
                    else:
                        neighbors.append(sentence[(pos + 1) % len(sentence)])
                    if pos > i:
                        neighbors.append(sentence[pos])
                    else:
                        neighbors.append(sentence[(neg - 1) % len(sentence)])
            else:
                neighbors = sentence
            M = 0
            best_ci = None
            for ci in target_synsets:
                max_sum = 0
                if ci is not None and target not in synsets.keys():
                    for other in neighbors:
                        if other is not target:
                            for cj in wn.synsets(other):
                                wup = wup_similarity_wordnet(ci, cj)
                                max_sum += wup
                    if max_sum > M:
                        M = max_sum
                        best_ci = ci
            if best_ci is not None:
                synset_name = best_ci.name()
                synsets[target] = synset_name
                embedding = node_embeddings[synset_name]
                sentence_node_embeddings.append(torch.tensor(embedding).to(device))
            else:
                sentence_node_embeddings.append(
                    torch.full([config.regularization.node_embedding_size], fill_value=1, dtype=torch.float).to(device))
                synsets[target] = None
        else:
            sentence_node_embeddings.append(
                torch.full([config.regularization.node_embedding_size], fill_value=1, dtype=torch.float).to(device))
            synsets[target] = None
    return torch.stack(sentence_node_embeddings), synsets


def wikidata_entities(word, retrieved_entities=None):
    if retrieved_entities and word in retrieved_entities and retrieved_entities[word]:
        return [retrieved_entities[word]]
    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=" + word + "&language=en&format=json"
    resp = requests.get(url=url, params={})
    if resp:
        try:
            entities_json = resp.json()
        except:
            return []
        if "success" in entities_json and entities_json["success"]:
            entities = entities_json["search"]
            return [e["id"] for e in entities]
        else:
            return []


async def wikidata_entities_async(session, word, retrieved_entities=None):
    # if already computed best_ci, keep it
    if retrieved_entities and word in retrieved_entities and retrieved_entities[word]:
        return [retrieved_entities[word]]
    # else ask
    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=" + word + "&language=en&format=json"
    try:
        async with session.get(url) as resp:
            data = await resp.read()
        # print(data.decode("utf-8"))
        # print(json.loads(data.decode("utf-8")))
        resp_dict = json.loads(data.decode("utf-8"))
        if "success" in resp_dict and resp_dict["success"]:
            entities = resp_dict["search"]
            return [e["id"] for e in entities]
        else:
            return []
    except Exception as ex:
        # print("FAILED:", ex)
        return []


def generate_entities_wup_list(ci, others, data_folder):
    with open(data_folder + "tsv/" + str(ci) + 'input.tsv', 'w') as f:
        f.write('q1\tq2\n')
        for y in others:
            f.write(ci + '\t' + y + '\n')
    f.close()


async def aggregate_similarity_wikidata(session, q1, q2, type="class"):
    url = "https://kgtk.isi.edu/similarity_api?q1=" + str(q1) + "&q2=" + str(q2) + "&similarity_type=" + type
    try:
        async with session.get(url) as resp:
            data = await resp.read()
        # print(data.decode("utf-8"))
        # print(json.loads(data.decode("utf-8")))
        resp_dict = json.loads(data.decode("utf-8"))
        if resp_dict and "similarity" in resp_dict:
            return resp_dict["similarity"]
        else:
            return 0
    except Exception as ex:
        # print("FAILED:", ex)
        return 0


async def wikidata_entity(target, i, sentence, node_embeddings, device, config, data_folder, evaluate_neighbors=False,
                          neighbors_d=2):
    sw = stopwords.words("english")
    retrieved_entities = {}
    sentence_node_embeddings = []

    target = target.replace(' ', '')
    stored_keys = retrieved_entities.keys()
    if "[CLS]" not in target and "[MASK]" not in target and "[PAD]" not in target and target.lower() not in sw:
        target_entities = wikidata_entities(target, retrieved_entities)
        target_entities = [s for s in target_entities if s in node_embeddings.keys()]
        if evaluate_neighbors and target not in stored_keys and target_entities is not None:
            neighbors = []
            for x in range(1, neighbors_d):
                neg = (i - x) % len(sentence)
                pos = (i + x) % len(sentence)
                if neg < i:
                    neighbors.append(sentence[neg])
                else:
                    neighbors.append(sentence[(pos + 1) % len(sentence)])
                if pos > i:
                    neighbors.append(sentence[pos])
                else:
                    neighbors.append(sentence[(neg - 1) % len(sentence)])
        else:
            neighbors = sentence

        neighbors = [x.replace(' ', '') for x in neighbors if x not in target]
        other_entities = await asyncio.gather(
            *[wikidata_entities_async(other, retrieved_entities) for other in neighbors])
        # Since max_sum is over all syntax meanings of all neighbours, i can flatten the result of asyn
        other_entities = [ent for neigh_other_ent in other_entities for ent in neigh_other_ent if ent is not None]

        best_ci = None
        session = aiohttp.ClientSession()
        tasks = []
        for ci in target_entities:
            tasks.append(async_bulk_similarity(session, target, ci, stored_keys, other_entities, data_folder))
        similarities = await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        if similarities:
            max_value = max(similarities)
            index_best = similarities.index(max_value)
            best_ci = target_entities[index_best]
        if best_ci is not None:
            retrieved_entities[target] = best_ci
            embedding = node_embeddings[best_ci]
            sentence_node_embeddings.append(embedding.to(device))
        else:
            sentence_node_embeddings.append(
                torch.full([config.regularization.node_embedding_size], fill_value=1, dtype=torch.float).to(device))
            retrieved_entities[target] = None
    else:
        sentence_node_embeddings.append(
            torch.full([config.regularization.node_embedding_size], fill_value=1, dtype=torch.float).to(device))
        retrieved_entities[target] = None
    return torch.stack(sentence_node_embeddings), retrieved_entities


def call_semantic_similarity(input_file, url):
    file_name = os.path.basename(input_file)
    files = {
        'file': (file_name, open(input_file, mode='rb'), 'application/octet-stream')
    }
    resp = requests.post(url, files=files, params={'similarity_types': 'all'})
    if resp:
        s = json.loads(resp.json())
        return pd.DataFrame(s)


async def async_bulk_similarity(session, target, ci, other_entities, data_folder):
    # print(ci, other_entities)
    if not os.path.exists(data_folder + "tsv/" + target + 'input.tsv'):
        generate_entities_wup_list(ci, other_entities, data_folder)
    max_sum = 0
    try:
        url = 'https://kgtk.isi.edu/similarity_api'
        data = FormData()
        data.add_field('file',
                       open(data_folder + "tsv/" + str(ci) + 'input.tsv', 'rb'),
                       filename=data_folder + "tsv/" + str(ci) + 'input.tsv',
                       content_type='text/tab-separated-values')
        async with session.post(url, data=data) as resp:
            data = await resp.read()
        resp = json.loads(json.loads(data.decode("utf-8")))
        for r in resp:
            # print(r)
            r = {k: 0 if not v else v for k, v in r.items()}
            max_sum += r['class'] + r['jc'] + r['text'] + r['topsim'] + r['transe']
        # print()
        return max_sum
    except Exception as ex:
        # print("FAILED:", ex)
        return max_sum
