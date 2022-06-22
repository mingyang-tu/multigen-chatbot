import torch
from utils.dictionary import Dictionary
from utils.seq_generator import SequenceGenerator
from utils.preprocessing import preprocess_concept

def MHGenerate(args, source_sentences, target_concepts, model, tokenizer, generator, device):
    data = preprocess_data(args, source_sentences, target_concepts, tokenizer)

    input_data = dict()
    for k, v in data.items():
        input_data[k] = torch.tensor(v).unsqueeze(dim=0).to(device)
    input_data["seq_generator"] = generator

    output_seq = model.generate(**input_data)

    return output_seq[0]

def preprocess_data(args, src, target_concepts, tokenizer, 
                    src_max_length=128, max_memory_size=400, max_triple_size=800):
    pad = tokenizer.encoder["<|pad|>"]
    eos = tokenizer.encoder["<|endoftext|>"]
    
    concept_dict = preprocess_concept(
        ' '.join(src), 
        target_concepts, 
        T=2, 
        max_B=getattr(args, 'max_num_B', 100), 
        max_concepts=getattr(args, 'max_concepts', 400), 
        max_triples=getattr(args, 'max_triples', 1000)
    )
    
    concept = concept_dict["concepts"]
    cpt_label = concept_dict["labels"]
    dist = concept_dict["distances"]
    relations = concept_dict["relations"]
    head_ids = concept_dict["head_ids"]
    tail_ids = concept_dict["tail_ids"]
    triple_labels = concept_dict["triple_labels"]
    relations = [x[0] for x in relations]

    assert(len(dist) == len(concept))

    concept_ids = []
    _concept_ids = []
    concept_mask = []
    bert_input = []
    bert_mask = []
    _concept_label = cpt_label.copy()
    head_ids_trunc = head_ids.copy()
    tail_ids_trunc = tail_ids.copy()
    relations_trunc = relations.copy()
    triple_labels_trunc = triple_labels.copy()
    _distance = dist.copy()
    vocab_map = [] # usage: cpt_prob.gather(-1, vocab_map) vocab_map size the same as gpt-2 vocab
    map_mask = [] # usage: cpt_prob_vocab.masked_fill_(map_mask == 0, 0)
    target_concept_ids = []

    distance = []
    concept_label = []
    count = 0
    for e, l, d in zip(concept, _concept_label, _distance):
        tok = tokenizer.encode(' ' + e)
        count += 1
        if len(tok) == 1:
            _concept_ids.append(tok[0])
            concept_ids.append(tok[0])
            distance.append(d)
            concept_label.append(l)
            if l == 1:
                target_concept_ids.append(tok[0])
        
    if len(concept_ids) > max_memory_size:
        concept_ids = concept_ids[:max_memory_size]
        concept_label = concept_label[:max_memory_size]
        distance = distance[:max_memory_size]

    while len(concept_ids) < max_memory_size:
        concept_ids.append(pad)
        concept_label.append(-1)
        distance.append(0)

    for idx in tokenizer.decoder.keys():
        try: 
            pos = _concept_ids.index(idx)
            vocab_map.append(pos)
            map_mask.append(1)
        except:
            vocab_map.append(0)
            map_mask.append(0)

    assert(len(vocab_map) == len(tokenizer.decoder)), len(vocab_map)
    assert(len(map_mask) == len(tokenizer.decoder)), len(map_mask)

    if len(head_ids_trunc) > max_triple_size:
        head_ids_trunc = head_ids_trunc[:max_triple_size]
        tail_ids_trunc = tail_ids_trunc[:max_triple_size]
        relations_trunc = relations_trunc[:max_triple_size]
        triple_labels_trunc = triple_labels_trunc[:max_triple_size]
        
    while len(head_ids_trunc) < max_triple_size:
        head_ids_trunc.append(0)
        tail_ids_trunc.append(0)
        relations_trunc.append(0)
        triple_labels_trunc.append(-1)

    src_input_ids = []
    for s in src:
        src_input_ids.extend(tokenizer.encode(' ' + s))
        src_input_ids.append(eos)
    src_position_ids = list(range(0, len(src_input_ids)))

    assert (len(src_input_ids) == len(src_position_ids))
    if len(src_input_ids) > src_max_length:
        src_input_ids = src_input_ids[:src_max_length]
        src_position_ids = src_position_ids[:src_max_length]

    attention_mask = [1] * len(src_input_ids)

    while len(src_input_ids) < src_max_length:
        src_input_ids += [pad]
        src_position_ids += [0]
        attention_mask += [0]
        
    assert(len(concept_ids) == max_memory_size), len(concept_ids)
    assert(len(distance) == max_memory_size), len(distance)

    output = {"src_input_ids": src_input_ids, 
              "attention_mask": attention_mask, 
              "src_position_ids": src_position_ids, 
              "concept_ids": concept_ids, 
              "concept_label": concept_label, 
              "distance": distance, 
              "head": head_ids_trunc, 
              "tail": tail_ids_trunc, 
              "relation": relations_trunc, 
              "triple_label": triple_labels_trunc, 
              "vocab_map": vocab_map, 
              "map_mask": map_mask}

    return output

def build_generator(args, tokenizer):
    generator = SequenceGenerator(
        args,
        Dictionary(tokenizer.encoder),
        tokenizer,
        beam_size=getattr(args, 'beam', 4),
        max_len_a=getattr(args, 'max_len_a', 0),
        max_len_b=getattr(args, 'max_len_b', 32),
        min_len=getattr(args, 'min_len', 1),
        normalize_scores=(not getattr(args, 'unnormalized', False)),
        len_penalty=getattr(args, 'lenpen', 1),
        unk_penalty=getattr(args, 'unkpen', 0),
        sampling=getattr(args, 'sampling', True),
        sampling_topk=getattr(args, 'sampling_topk', 80),
        sampling_topp=getattr(args, 'sampling_topp', 0.95),
        temperature=getattr(args, 'temperature', 1.),
        diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
        diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len=getattr(args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
    )  
    return generator