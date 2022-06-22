import spacy
import networkx as nx
import json
import random

#################### ground_concepts_simple.py ####################
blacklist = set(["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

with open("./data/concept.txt", "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])

vocab_dict = json.loads(open("./data/vocab.json", 'r').readlines()[0])
model_vocab = []
for tok in vocab_dict.keys():
    model_vocab.append(tok[1:])

# match the concepts of the input sentence
def hard_ground(sent):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist and t.lemma_ in model_vocab:
            if t.pos_ == "NOUN" or t.pos_ == "VERB":
                res.add(t.lemma_)
    return res

def match_mentioned_concepts(sent):
    question_concepts = hard_ground(sent)

    return list(question_concepts)
    

#################### find_neighbours.py ####################
concept2id = {}
id2concept = {}
with open("./data/concept.txt", "r", encoding="utf8") as f:
    for w in f.readlines():
        concept2id[w.strip()] = len(concept2id)
        id2concept[len(id2concept)] = w.strip()
id2relation = {}
relation2id = {}
with open("./data/relation.txt", "r", encoding="utf8") as f:
    for w in f.readlines():
        id2relation[len(id2relation)] = w.strip()
        relation2id[w.strip()] = len(relation2id)

cpnet = nx.read_gpickle("./data/cpnet.graph")
cpnet_simple = nx.Graph()
for u, v, data in cpnet.edges(data=True):
    w = data['weight'] if 'weight' in data else 1.0
    if cpnet_simple.has_edge(u, v):
        cpnet_simple[u][v]['weight'] += w
    else:
        cpnet_simple.add_edge(u, v, weight=w)

with open("./data/total_concepts.txt", 'r', encoding="utf8") as f:
    total_concepts_id = [concept2id[x.strip()] for x in list(f.readlines())]
total_concepts_id_set = set(total_concepts_id)

def get_edge(src_concept, tgt_concept):
    try:
        rel_list = cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]["rel"] for item in rel_list]))
    except:
        return []

def find_neighbours_frequency(source_concepts, target_concepts, T, max_B):
    """
    T: maximum distance
    max_B: maximum number of nodes found per BFS
    """
    source = [concept2id[s_cpt] for s_cpt in source_concepts]
    start = source
    Vts = dict([(x,0) for x in start])
    Ets = {}
    for t in range(T):
        V = {}
        for s in start:
            if s in cpnet_simple:
                for n in cpnet_simple[s]:
                    if n not in Vts and n in total_concepts_id_set:
                        if n not in Vts:
                            if n not in V:
                                V[n] = 1
                            else:
                                V[n] += 1

                        if n not in Ets:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n] = {s: rels}  
                        else:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n].update({s: rels})  

        V = list(V.items())
        # keep those points with higher frequency
        count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B]
        start = [x[0] for x in count_V if x[0] in total_concepts_id_set]
        
        Vts.update(dict([(x, t+1) for x in start]))
    
    _concepts = list(Vts.keys())
    _distances = list(Vts.values())
    concepts = []
    distances = []
    for c, d in zip(_concepts, _distances):
        concepts.append(c)
        distances.append(d)
    assert(len(concepts) == len(distances))
    
    triples = []
    for v, N in Ets.items():
        if v in concepts:
            for u, rels in N.items():
                if u in concepts:
                    triples.append((u, rels, v))
    
    nearest_concept = []
    min_distance = 0
    labels = []
    found_num = 0
    for idx, c in enumerate(concepts):
        if c in target_concepts:
            if (found_num < 1) or (distances[idx] == min_distance):
                nearest_concept.append(id2concept[c].replace("_", " "))
                min_distance = distances[idx]
            found_num += 1
            labels.append(1)
        else:
            labels.append(0)
    
    res = [id2concept[x].replace("_", " ") for x in concepts]
    triples = [(id2concept[x].replace("_", " "), y, id2concept[z].replace("_", " ")) for (x,y,z) in triples]

    return {"concepts":res, "labels":labels, "distances":distances, "triples":triples}, nearest_concept
    """
    concepts: node found through BFS
    labels: if (node in concepts == node in tgt concepts): label = 1
    distances: number of edges from that node to src node
    triples: (start position, weight of edge, end position) of all edges
    """

#################### filter_triple.py ####################
def bfs(start, triple_dict, source):
    paths = [[[start]]]
    stop = False
    shortest_paths = []
    count = 0
    while 1:
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            if triple_dict.get(path[-1], False):
                triples = triple_dict[path[-1]]
                for triple in triples:
                    new_paths.append(path + [triple[0]])

        for path in new_paths:
            if path[-1] in source:
                stop = True
                shortest_paths.append(path)
        
        if (count == 2):    #####
            break
        paths.append(new_paths)
        count += 1
    return shortest_paths

def filter_directed_triple(data, max_concepts, max_triples):
    max_neighbors = 5
    
    triple_dict = {}
    triples = data['triples']
    concepts = data['concepts']
    labels = data['labels']
    distances = data['distances']

    for t in triples:
        head, tail = t[0], t[-1]
        head_id = concepts.index(head)
        tail_id = concepts.index(tail)
        if distances[head_id] <= distances[tail_id]:    # forward edges
            if t[-1] not in triple_dict:
                triple_dict[t[-1]] = [t]
            else:
                if len(triple_dict[t[-1]]) < max_neighbors:
                    triple_dict[t[-1]].append(t)

    starts = []
    for l, c in zip(labels, concepts):
        if l == 1:
            starts.append(c)

    sources = []
    for d, c in zip(distances, concepts):
        if d == 0:
            sources.append(c)

    shortest_paths = []
    for start in starts:
        shortest_paths.extend(bfs(start, triple_dict, sources))
    
    ground_truth_triples = []
    for path in shortest_paths:
        for i, n in enumerate(path[:-1]):
            ground_truth_triples.append((n, path[i+1]))

    ground_truth_triples_set = set(ground_truth_triples)

    _triples = []
    triple_labels = []
    for k,v in triple_dict.items():
        for t in v:
            _triples.append(t)
            if (t[-1], t[0]) in ground_truth_triples_set:
                triple_labels.append(1)
            else:
                triple_labels.append(0)

    if max_concepts != None:
        concepts = concepts[:max_concepts]
    if max_triples != None:
        _triples = _triples[:max_triples]
        triple_labels = triple_labels[:max_triples]

    heads = []
    tails = []
    for triple in _triples:
        heads.append(concepts.index(triple[0]))
        tails.append(concepts.index(triple[-1]))

    data['relations'] = [x[1] for x in _triples]
    data['head_ids'] = heads
    data['tail_ids'] = tails
    data['triple_labels'] = triple_labels
    data.pop('triples')
    
    return data

def preprocess_concept(source_sentence, target_concept, T=2, max_B=100, max_concepts=400, max_triples=1000):
    concepts_nv = match_mentioned_concepts(source_sentence)
    target_cpt_ids = [concept2id[t_cpt] for t_cpt in target_concept]
    e, _ = find_neighbours_frequency(concepts_nv, target_cpt_ids, T, max_B)
    data_processed = filter_directed_triple(e, max_concepts, max_triples)
    return data_processed

with open("./data/target_concepts.txt", 'r', encoding="utf8") as f:
    main_concepts = [x.strip() for x in list(f.readlines())]

print(f"Number of Target Concepts: {len(main_concepts)}")
print(main_concepts)

main_cpt_ids = [concept2id[t_cpt] for t_cpt in main_concepts]


def find_nearest_concept(source_sentence, T=2, max_B=100):
    concepts_nv = match_mentioned_concepts(source_sentence)
    _, nearest_concept = find_neighbours_frequency(concepts_nv, main_cpt_ids, T, max_B)

    if len(nearest_concept) > 1:
        nearest_concept = random.sample(nearest_concept, 2)
        
    print(f" >>> Target concepts: {nearest_concept}")
    return nearest_concept

with open("keywords.json", "r") as f:
    keywords = json.load(f)
for key, val in keywords.items():
    # separate words by its length (one, others)
    one_lemma = []
    multi_lemma = []
    for word in val:
        split = [token.lemma_ for token in nlp(word)]
        if len(split) >= 2:
            multi_lemma.append(" ".join(split))
        else:
            one_lemma.append(split[0])
        keywords[key] = [one_lemma, multi_lemma]

def find_hit(text):
    lemma_utterance = [token.lemma_ for token in nlp(text)]
    for key, (one, multi) in keywords.items():
        intersection = set(one) & set(lemma_utterance)
        # check whether the word, the length is bigger than 2, is in the utterance
        for m in multi:
            unsplit_utterance = " ".join(lemma_utterance)
            if m in unsplit_utterance:
                intersection.add(m)
        if len(intersection) != 0:
            print("Hit!!!")
            return True
    return False