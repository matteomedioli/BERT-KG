import torch
from itertools import zip_longest
from tqdm import tqdm
import json

def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")

def freebase2wikidata(entities):
    """
    This method constructs a dictionary mapping an freebase id to some wikidata entities.


    :param entities: an iterable of string entities
    :return:
    """
    import requests
    from SPARQLWrapper import SPARQLWrapper, JSON
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    def dbpedia_with_freebase(entities):
        """
        :param entities: list of entities
        :return: dict: { "freebase" : { "wikidata1" : {},
                                        "wikidata2" : {},
                                      },
                        ...}
        """
        ### Part 1 ####
        # Query DBPedia for Wikidata Ids

        # finds all wikidata_ids that have this freebase id
        dbpedia_query = """PREFIX dbpedia: <http://dbpedia.org/resource/>
        SELECT DISTINCT ?other WHERE {
            ?obj (owl:sameAs) <http://rdf.freebase.com/ns/%s>.
            ?obj (owl:sameAs) ?other .
            FILTER (strstarts(str(?other), 'http://www.wikidata.org/entity/'))
        }"""

        res = {}
        for e in entities:
            q = dbpedia_query % e[1:].replace('/', '.')  # /m/xxxx -> m.xxxx
            sparql.setQuery(q)
            results = sparql.query().convert()

            for result in results["results"]["bindings"]:
                if e not in res:
                    res[e] = {}

                wd = result['other']['value'].replace(
                    'http://www.wikidata.org/entity/', '')
                res[e][wd] = {}
        return res

    def wikidata_with_freebase(entities):
        """

        :param entities: list of freebase entities
        :return: dict {
                      '/m/01bs9f': {'Q13582652': {'alternatives': set(),
                                                  'description': 'engineer specialising
                                                                  in design, construction
                                                                  and maintenance of the
                                                                  built environment',
                                                  'label': 'civil engineer',
                                                  'wikipedia': set()
                                                  }
                                   },
                     '/m/01cky2': ...
                     }
        """
        query_wikidata_with_freebase = '''
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?wd ?fb ?wdLabel ?wdDescription ?alternative ?sitelink
        WHERE {
          ?wd wdt:P646 ?fb .
          OPTIONAL { ?wd schema:description ?itemdesc . }
          OPTIONAL { ?wd skos:altLabel ?alternative . 
                       FILTER (lang(?alternative) = "en").
                     }
          OPTIONAL { ?sitelink schema:about ?wd . 
                       ?sitelink schema:inLanguage "en" .
                       FILTER (SUBSTR(str(?sitelink), 1, 25) = "https://en.wikipedia.org/") .
                     } .
          VALUES ?fb { "%s" }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

        }'''
        url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        res = {}

        for ents in zip(*(iter(entities),) * 100):
            query_ = query_wikidata_with_freebase % '" "'.join(ents)
            data = requests.get(url, params={'query': query_, 'format': 'json'}).json()
            for item in data['results']['bindings']:
                wd = item['wd']['value'].replace('http://www.wikidata.org/entity/', '')
                fb = item['fb']['value']
                label = item['wdLabel']['value'] if 'wdLabel' in item else None
                desc = item['wdDescription']['value'] if 'wdDescription' in item else None
                alias = {item['alternative']['value']} if 'alternative' in item else set()
                sitelink = {item['sitelink']['value']} if 'sitelink' in item else set()

                if fb not in res:
                    res[fb] = {}

                if wd not in res[fb]:
                    res[fb][wd] = {'label': label,
                                   'description': desc,
                                   'wikipedia': sitelink,
                                   'alternatives': alias}

                res[fb][wd]['wikipedia'] |= sitelink
                res[fb][wd]['alternatives'] |= alias
        return res

    def wikidata_with_wikidata(entities):
        """

        :param dict entities: { "freebase" : { "wikidata1" : {},
                                        "wikidata2" : {},
                                      },
                        ...}
        :return: dict {
                      '/m/01bs9f': {'Q13582652': {'alternatives': set(),
                                                  'description': 'engineer specialising
                                                                  in design, construction
                                                                  and maintenance of the
                                                                  built environment',
                                                  'label': 'civil engineer',
                                                  'wikipedia': set()
                                                  }
                                   },
                     '/m/01cky2': ...
                     }
        """
        query_wd_with_wd = '''PREFIX wikibase: <http://wikiba.se/ontology#>
                   PREFIX wd: <http://www.wikidata.org/entity/>
                   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?wd ?fb ?wdLabel ?wdDescription ?alternative ?sitelink
            WHERE {
              BIND(wd:%s AS ?wd).
              OPTIONAL { ?wd schema:description ?itemdesc . }
              OPTIONAL { ?wd skos:altLabel ?alternative . 
                           FILTER (lang(?alternative) = "en").
                         }
              OPTIONAL { ?sitelink schema:about ?wd . 
                           ?sitelink schema:inLanguage "en" .
                           FILTER (SUBSTR(str(?sitelink), 1, 25) = "https://en.wikipedia.org/") .
                         } .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

            }'''
        url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'

        res = {}
        for fb, wd_ids in entities.items():
            for wd_id in wd_ids:
                query_ = query_wd_with_wd % wd_id
                data = requests.get(url,
                                    params={'query': query_, 'format': 'json'}).json()
                for item in data['results']['bindings']:
                    wd = item['wd']['value'].replace('http://www.wikidata.org/entity/',
                                                     '')
                    # fb = item['fb']['value']
                    label = item['wdLabel']['value'] if 'wdLabel' in item else None
                    desc = item['wdDescription'][
                        'value'] if 'wdDescription' in item else None
                    alias = {
                        item['alternative']['value']} if 'alternative' in item else set()
                    sitelink = {
                        item['sitelink']['value']} if 'sitelink' in item else set()

                    if fb not in res:
                        res[fb] = {}

                    if wd not in res[fb]:
                        res[fb][wd] = {'label': label,
                                       'description': desc,
                                       'wikipedia': sitelink,
                                       'alternatives': alias}

                    res[fb][wd]['wikipedia'] |= sitelink
                    res[fb][wd]['alternatives'] |= alias
        return res

    # lets first try to find the freebase entities in wikidata
    result = wikidata_with_freebase(entities)
    logging.info("Found %s freebase entities in wikidata (from total %s)." %
                 (len(result), len(entities)))

    # then find the remaining ids in dbpedia
    missing_entities = set(entities) - set(result.keys())
    result_missing = dbpedia_with_freebase(missing_entities)

    # and query the wikidata information afterwards
    result_missing = wikidata_with_wikidata(result_missing)
    logging.info("Found %s missing entities via dbpedia in wikidata (from total %s "
                 "missing entities)." %
                 (len(result_missing), len(missing_entities)))

    # merge the two dicts
    result = {**result, **result_missing}
    # and remove the sets
    for fb, wds in result.items():
        for wd_id, stats in wds.items():
            result[fb][wd_id]['wikipedia'] = stats['wikipedia'].pop() if stats[
                'wikipedia'] else None
            result[fb][wd_id]['alternatives'] = list(stats['alternatives'])

    logging.info("Final: Found %s freebase entities in wikidata (from total %s)." %
                 (len(result), len(entities)))

    return result

def wn18rr_node_dict():
    D = {}
    for file_name in ["train", "valid", "test"]:
        id_file = "data/WN18RR/" + file_name + ".txt"
        name_file = "data/WN18RR/" + file_name + "_text.txt"
        files = [open(filename) for filename in [id_file, name_file]]
        for lines in tqdm(zip_longest(*files, fillvalue='')):
            id_triple = lines[0].replace("\n", "").split("\t")
            name_triple = lines[1].replace("\n", "").split("\t")
            if id_triple[0] not in D:
                D[id_triple[0]] = name_triple[0]
            if id_triple[2] not in D:
                D[id_triple[2]] = name_triple[2]
    lines = tuple(open("data/WN18RR/entity2id.txt", 'r'))
    node_dict = {}
    gat_embeddings = torch.load("data/WN18RR/entities_gat.pt") # TODO: write entities tensor in the right place and read here
    for line in tqdm(lines):
        node_dict[D[line.split()[0]]] = gat_embeddings[int(line.split()[1])]
    return node_dict

def fb15k_node_dict():
    with open('data/FB15K-237/entity2wikidata.json') as json_file:
        entity2wikidata = json.load(json_file)
    lines = tuple(open("data/FB15K-237/entity2id.txt", 'r'))
    node_dict = {}
    gat_embeddings = torch.load("data/FB15K-237/entities_gat.pt") # TODO: write entities tensor in the right place and read here
    for line in tqdm(lines):
        fb_id = line.split()[0]
        if fb_id in entity2wikidata:
            node_dict[entity2wikidata[fb_id]["wikidata_id"]] = gat_embeddings[int(line.split()[1])]
    return node_dict