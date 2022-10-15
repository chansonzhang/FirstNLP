from functools import partial
from typing import Iterable


import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import torch


# Add LanguageDetector and assign it a string name
@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)


def _new_spacy_pipeline(name, disable, post_process_func):
    if not disable:
        nlp = spacy.load(name)
    else:
        nlp = spacy.load(name, disable=disable)
    if post_process_func:
        nlp = post_process_func(nlp)
    return nlp


def _modify_suffix_search(nlp):
    suffixes = nlp.Defaults.suffixes + [r'''\w+-\w+''',
                                        r'''\w+&\w+''',
                                        r'''\w+-\w+-\w+''',
                                        r'''\w+\.\w+\.''',
                                        r'''\d+\.\d+''',
                                        r'''\d+\.\d+\.\d+''',
                                        r'''\d+\,\d+''',
                                        r'''[\d\w]+''']
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    return nlp


def _add_language_dector_to_pipeline(nlp):
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('language_detector', last=True)
    return nlp


# name -> [model_name, disable, post_process_func, instance]
_loaded_instances = {
    "get_spacy_use": ['en_use_md', None, None, None],
    "get_spacy_base": ['en_core_web_md', None, None, None],
    "get_spacy_zh_base": ['zh_core_web_sm', None, None, None],
    "get_spacy_md_no_ner": ['en_core_web_md', ['ner'], _modify_suffix_search, None],
    "get_spacy_lite": ['en_core_web_md', ['parser', 'ner',], None, None],
    "get_spacy_lite_sm": ['en_core_web_sm', ['parser', 'ner',], None, None],
    "get_spacy_tokenizer_only_md": ['en_core_web_md', ['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer',], None, None],
    "get_spacy_tokenizer_only_sm": ['en_core_web_sm', ['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer',], None, None],
    "get_spacy_with_language_detector": ['en_core_web_md',
                                         ['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'],
                                         _add_language_dector_to_pipeline,
                                         None],
}


def _new_or_return(name, create_new=False):
    global _loaded_instances
    model_name, disable, post_process_func, instance = _loaded_instances[name]
    if not create_new and instance:
        return instance

    instance = _new_spacy_pipeline(model_name, disable, post_process_func)
    if not create_new:
        _loaded_instances[name][3] = instance

    return instance


for k in _loaded_instances.keys():
    globals()[k] = partial(_new_or_return, k)


def get_spacy_use():
    nlp = spacy.load('en_use_md')
    return nlp

#
# def get_spacy_base():
#     nlp = spacy.load('en_core_web_md')
#     return nlp
#
#
# def get_spacy_zh_base():
#     nlp = spacy.load('zh_core_web_sm')
#     return nlp
#
#
# def get_spacy_md_no_ner():
#     nlp = spacy.load('en_core_web_md', disable=['ner'])
#     suffixes = nlp.Defaults.suffixes + [r'''\w+-\w+''',
#                                         r'''\w+&\w+''',
#                                         r'''\w+-\w+-\w+''',
#                                         r'''\w+\.\w+\.''',
#                                         r'''\d+\.\d+''',
#                                         r'''\d+\.\d+\.\d+''',
#                                         r'''\d+\,\d+''',
#                                         r'''[\d\w]+''']
#     suffix_regex = spacy.util.compile_suffix_regex(suffixes)
#     nlp.tokenizer.suffix_search = suffix_regex.search
#     return nlp
#
#
# def get_spacy_lite():
#     nlp = spacy.load('en_core_web_md', disable=['parser', 'ner',])
#     return nlp
#
#
# def get_spacy_lite_sm():
#     nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner',])
#     return nlp
#
# #
# # disable it, by default, we don't use it, because the lg model is 10x large
# # def get_spacy_lite_lg():
# #     nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner',])
# #     return nlp
#
#
# def get_spacy_tokenizer_only_md():
#     nlp = spacy.load('en_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer',])
#     return nlp
#
#
# def get_spacy_tokenizer_only_sm():
#     nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer',])
#     return nlp
#
#
# def get_spacy_with_language_detector():
#     nlp = spacy.load('en_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
#     nlp.add_pipe('sentencizer')
#     nlp.add_pipe('language_detector', last=True)
#     return nlp
#


def get_spacy_stop_words():
    return spacy.lang.en.stop_words.STOP_WORDS


def spacy_encoder_for_torch(nlp, docs: Iterable[str]) -> torch.Tensor:
    return torch.stack([torch.from_numpy(x.vector) for x in nlp.pipe(docs)])