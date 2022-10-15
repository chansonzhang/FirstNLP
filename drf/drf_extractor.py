from typing import List, Dict, Any
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from lib.text.util import TextProcessor
from sklearn.metrics import mutual_info_score
from collections import OrderedDict
import numpy as np
import json


@dataclass_json
@dataclass
class Domain:
    domain_name: str
    corpus: List[str]


tp = TextProcessor()


def get_ngram_tokens(text: str):
    return tp.get_ngram_tokens(text,
                               to_filter_stopword=True,
                               to_lemma=False,
                               to_lower=True,
                               to_clean_token=True)


def extract_drf(domains: List[Domain], top_n=10, rho=1.0) -> Dict[str, List[str]]:
    domain_grams: Dict[str, List[Any]] = {}
    grams_map: Dict[str, List[Any]] = {}
    all_grams: OrderedDict[str, Any] = OrderedDict()
    for d in domains:
        d_name = d.domain_name
        domain_grams[d_name] = []
        for text in d.corpus:
            unigrams, bigrams, _, _ = get_ngram_tokens(text=text)
            grams = unigrams + bigrams
            domain_grams[d_name].extend(grams)
            grams_map[text] = list(set(grams))
            for g in grams:
                all_grams[g] = g

    gram_to_id: Dict[str, int] = {g: _id for _id, g in enumerate(all_grams.keys())}

    gram_count_matrix: List[List[int]] = []
    for d in domains:
        for text in d.corpus:
            gram_counts: List[int] = [0] * len(all_grams)
            grams = grams_map[text]
            for g in grams:
                gram_counts[gram_to_id[g]] += 1

            gram_count_matrix.append(gram_counts)

    np_gram_count_matrix = np.array(gram_count_matrix)

    drfs: Dict[str, List[str]] = {}

    sample_id_offset = 0
    for d in domains:
        labels: List[int] = []
        for dd in domains:
            label = 1 if dd.domain_name == d.domain_name else 0
            labels += [label] * len(dd.corpus)

        try:
            gram_mis = [mutual_info_score(labels_true=labels,
                                          labels_pred=np_gram_count_matrix[:, j])
                        for j in range(len(all_grams.keys()))]
        except Exception as e:
            print()

        sorted_gram_id_by_mi = sorted(range(0, len(all_grams.keys())), key=lambda j: gram_mis[j])
        count_in_domain: int = lambda gid: np.sum(
            np_gram_count_matrix[sample_id_offset:sample_id_offset + len(d.corpus), gid])
        count_in_other_domains: int = lambda gid: np.sum(np_gram_count_matrix[:, gid]) - count_in_domain(gid)

        sorted_gram_id_by_mi_filtered = [gid for gid in sorted_gram_id_by_mi
                                         if count_in_domain(gid) > 0 and count_in_other_domains(gid) / count_in_domain(
                gid) <= rho]
        top_n_mi_grams: List[str] = [list(all_grams.keys())[gram_id]
                                     for gram_id in
                                     sorted_gram_id_by_mi_filtered[-top_n:]]

        drfs[d.domain_name] = top_n_mi_grams
        sample_id_offset += len(d.corpus)

    return drfs


if __name__ == '__main__':
    from lib.nlp.spacy_util import get_spacy_use

    nlp = get_spacy_use()


    def split_to_sentences(text: str) -> List[str]:
        return [x.text for x in nlp(text).sents]


    domains: List[Domain] = [
        Domain(domain_name='Walden Pond', corpus=split_to_sentences("""
        Walden Pond is a pond in Concord, Massachusetts, in the United States. A famous example of a kettle hole, it was formed by retreating glaciers 10,000–12,000 years ago.[4] The pond is protected as part of Walden Pond State Reservation, a 335-acre (136 ha) state park and recreation site managed by the Massachusetts Department of Conservation and Recreation.[1] The reservation was designated a National Historic Landmark in 1962 for its association with the writer Henry David Thoreau (1817–1862), whose two years living in a cabin on its shore provided the foundation for his famous 1854 work, Walden; or, Life in the Woods. The National Historic Preservation Act of 1966 ensured federal support for the preservation of the pond.[5]
        """)),
        Domain(domain_name='China', corpus=split_to_sentences("""
        China,[i] officially the People's Republic of China (PRC),[j] is a country in East Asia. It is the world's most populous country with a population exceeding 1.4 billion people.[k] China spans five geographical time zones[l] and borders fourteen countries by land,[m] the most of any country in the world, tied with Russia. China also has a narrow maritime boundary with the disputed Taiwan.[n][o] Covering an area of approximately 9.6 million square kilometers (3,700,000 sq mi), it is the world's third largest country by total land area.[p] The country consists of 23 provinces,[n] five autonomous regions, four municipalities, and two Special Administrative Regions (Hong Kong and Macau). The national capital is Beijing, and the most populous city and financial center is Shanghai.
        """))
    ]

    print(json.dumps(extract_drf(domains=domains), indent=4))
