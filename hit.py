import argparse
import json
from collections import defaultdict

import spacy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction",
        default="output.jsonl",
        type=str,
        help="file that save the prediction dialogs",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    with open("keywords.json") as f:
        keywords = json.load(f)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # lemmatize words in keywords
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

    with open(args.prediction, "r") as f:
        dialog = [json.loads(line) for line in f]

    statistics = []
    hit_counts = {key: defaultdict(int) for key in keywords.keys()}
    hit_num = 0
    for d in tqdm(dialog):
        # start with the second utterance from the simulator
        for index in range(2, len(d["dialog"]), 2):
            lemma_utterance = [token.lemma_ for token in nlp(d["dialog"][index])]
            service_hits = defaultdict(int)
            for key, (one, multi) in keywords.items():
                intersection = set(one) & set(lemma_utterance)
                # check whether the word, the length is bigger than 2, is in the utterance
                for m in multi:
                    unsplit_utterance = " ".join(lemma_utterance)
                    if m in unsplit_utterance:
                        intersection.add(m)
                service_hits[key] += len(intersection)
                statistics += list(intersection)
                for hit in intersection:
                    hit_counts[key][hit] += 1
            # Is there a keyword in this utterance
            isService = sum(service_hits.values()) != 0
            if isService:
                hit_num += 1
                break

    print(f"total dialogs: {len(dialog)}")
    print("hit count by service:")
    for service, counts in hit_counts.items():
        print(f"\tservice: {service}")
        for keyword, hit in counts.items():
            print(f"\t\t{keyword} {hit}")
    print(f"hit rate: {hit_num/len(dialog):.3f}")
