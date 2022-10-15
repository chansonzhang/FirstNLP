import copy
import re
from functools import reduce
from typing import List, Tuple

import markdown
from bs4 import BeautifulSoup
from spacy.tokens import Doc
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

from lib.nlp.spacy_util import get_spacy_lite, get_spacy_tokenizer_only_md
from lib.text.util_data import (
    _CONTRACTION_PATTERNS,
    _Q_HEAD_RE,
    _Q_TAIL_RE,
    _STOP_WORDS,
    _SPACE_END_MARK_RE,
    _SPECIAL_ABBREVIATIONS,
    _UNICODE_RE
)
import logging

logger = logging.getLogger(__name__)


class TextProcessor(object):
    _REPLACE_RE = re.compile(r'([^\w\n,.?:;"\']| )')
    _SEGMENT_RE = re.compile(r'(\n|[,.?:;"\'] | ["\'])')
    _SPECIAL_CHAR_RE = re.compile(r'[,.?:;"\'*]')

    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if not TextProcessor.__initialized:
            self._nlp_with_lemma = get_spacy_lite()  # get_spacy_base()
            self._nlp_no_lemma = get_spacy_tokenizer_only_md()
            self._stemmer = PorterStemmer()
            TextProcessor.__initialized = True

    def get_clean_tokens(self, text: str) -> List[str]:
        """
        Clean up and tokenize the text for most downstream usage. for example compute sentence embedding.
        1. remove noises
        2. do not filter stopwords
        3. do not do lemmatization
        4. lower the string
        5. clean up token

        Typically, you may want to call get_clean_str() directly to get the "clean" text.

        If you are intended to perform token based operation, calling get_ngram_token() is better choice

        @param text:
        @return:
        """
        unigrams, _, _, _ = self.get_ngram_tokens(text, to_filter_stopword=False,
                                                  to_lemma=False,
                                                  to_lower=True,
                                                  to_clean_token=True)
        return unigrams

    def get_clean_str(self, text: str) -> str:
        """
        Convenient method to get "clean" string for most scenarios that need string instead of tokens, for example,
        computing sentence-level embedding.
        @param text:
        @return:
        """
        tokens = self.get_clean_tokens(text)
        return " ".join([x for x in tokens if x])

    def get_normalized_tokens(self, text: str) -> List[str]:
        """
        Clean up and tokenize the text for keywords based processing. for example compute normalized keywords.
        1. remove noises
        2. filter stopwords
        3. lemmatization
        4. lower the string
        5. clean up token

        Typically, you may want to call get_normalized_str() directly to get the "normalized" text.

        If you are intended to perform token based operation, calling get_ngram_token() is better choice

        @param text:
        @return:
        """
        unigrams, _, _, _ = self.get_ngram_tokens(text, to_filter_stopword=True,
                                                  to_lemma=True,
                                                  to_lower=True,
                                                  to_clean_token=True)
        return unigrams

    def get_normalized_str(self, text: str) -> str:
        """
        Convenient method to get "normalized" string for keywords based algorithms, for example, normalize keywords
        @param text:
        @return:
        """
        tokens = self.get_normalized_tokens(text)
        return " ".join([x for x in tokens if x])

    @classmethod
    def remove_special_characters(cls, text):
        """
        TODO: to take it private
        @param text:
        @return:
        """
        if not text:
            return ""
        return cls._SPECIAL_CHAR_RE.sub("", text)

    def get_normalized_ngram_tokens(self, text: str
                                    ) -> Tuple[List[str], List[str]]:
        """
        return uni-gram and bi-gram tokens for most n-gram token based retrieval algorithm.
        For example, doc-token retrieval and topicality score.

        1. remove stop words
        2. turn on lemmatization
        3. to lower
        4. clean up tokens

        @param text:
        @return:
        """
        unigrams, bigrams, _, _ = self.get_ngram_tokens(text,
                                                        to_filter_stopword=True,
                                                        to_lemma=True,
                                                        to_lower=True,
                                                        to_clean_token=True)
        return unigrams, bigrams

    def get_ngram_tokens(self, text: str,
                         to_filter_stopword: bool = False,
                         to_lemma: bool = False,
                         to_lower: bool = True,
                         to_clean_token: bool = True,
                         include_original_unigram: bool = False,
                         include_stemming_bigram: bool = False,
                         ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        tokenize the text for token based retrieval purpose, return uni-gram list and bi-gram list
        @param text: text doc
        @param to_filter_stopword: whether to filter stopwords
        @param to_lemma: whether to use lemma
        @param to_lower: whether to lower case
        @param to_clean_token: clear out special characters from token
        @param include_original_unigram: include unigram tokens in original form excluding stop-words
        @param include_stemming_bigram: include bigram tokens in stemming form including stop-words
        @return: [[unigram], [bigram], [unigram in original form], [stemming bigram]
        """

        if to_lemma:
            nlp = self._nlp_with_lemma
        else:
            nlp = self._nlp_no_lemma
        to_stem = include_stemming_bigram

        text = text or ''
        clean_text = self._REPLACE_RE.sub(' ', text)
        segments = self._SEGMENT_RE.split(clean_text)  # segment first to avoid computing n-gram across certain boundary

        def clean_and_append(s, l):
            if to_clean_token:
                s = self.remove_special_characters(s).strip()
            if s:
                l.append(s)

        unigrams = []
        bigrams = []
        orig_unigrams = []
        stem_bigrams = []
        for seg in segments:
            if to_lower:
                seg = seg.lower()

            nlp_tokens = nlp(seg)

            # normalized tokens
            tokens = []
            for x in nlp_tokens:
                if x.is_space or x.is_punct or (to_filter_stopword and x.is_stop):
                    continue
                if to_lemma:
                    s = x.lemma_
                else:
                    s = x.text
                clean_and_append(s, tokens)

            # orig tokens
            orig_tokens = []
            if include_original_unigram:
                for x in nlp_tokens:
                    if x.is_space or x.is_punct or (to_filter_stopword and x.is_stop):
                        continue
                    clean_and_append(x.text, orig_tokens)

            #  stem tokens
            stemming_tokens = []
            if to_stem:
                for x in nlp_tokens:
                    if x.is_space or x.is_punct:
                        continue
                    s = self._stemmer.stem(x.text)
                    clean_and_append(s, stemming_tokens)

            unigrams += tokens
            if len(tokens) > 1:
                bigrams += [' '.join(x) for x in zip(tokens[:-1], tokens[1:])]

            orig_unigrams += orig_tokens
            if len(stemming_tokens) > 1:
                stem_bigrams += [' '.join(x) for x in zip(stemming_tokens[:-1], stemming_tokens[1:])]

        return unigrams, bigrams, orig_unigrams, stem_bigrams

    @property
    def stemmer(self):
        return self._stemmer

    @property
    def REPLACE_RE(self):
        return self._REPLACE_RE

    @property
    def SPECIAL_CHAR_RE(self):
        return self._SPECIAL_CHAR_RE


_CLEAN_RE = re.compile(r'[^\w\n.,]')
_COLLAPSE_RE = re.compile(r' +')
_BREAK_RE = re.compile(r'[\n.]')


def split_long_body_text(body: str, size: int) -> List[str]:
    """
    DEPRECATED: Prefer to use prepare md_doc

    Clean up and split long body text into smaller chunks. Honor sentence boundary.
    """
    body = body or ''
    sentences = [x.strip() for x in
                 _BREAK_RE.split(_COLLAPSE_RE.sub(' ', _CLEAN_RE.sub(' ', body))) if x.strip()]
    out = []
    curr = []
    curr_len = 0
    for s in sentences:
        curr_len += len(s)
        curr.append(s)
        if curr_len > size:
            out.append('. '.join(curr))
            curr = []
            curr_len = 0
    if curr:
        out.append('. '.join(curr))
    return out


def split_long_body_text_by_tokens(body: str, n_tokens: int) -> List[str]:
    """
    DEPRECATED: Prefer to use prepare md_doc

    Clean up and split long body text into smaller chunks based on
    the number of tokens. Honor sentence boundary.
    """
    body = body or ''
    sentences = [x.strip() for x in
                 _BREAK_RE.split(_COLLAPSE_RE.sub(' ', _CLEAN_RE.sub(' ', body))) if x.strip()]
    out = []
    curr = []
    curr_tokens = 0
    for s in sentences:
        curr_tokens += len(s.split())
        curr.append(s)
        if curr_tokens >= n_tokens:
            out.append('. '.join(curr))
            curr = []
            curr_tokens = 0
    if curr:
        out.append('. '.join(curr))
    return out


def split_long_body_text_to_shingles(body: str, shingle_size: int) -> List[str]:
    """
    Turn long body text to overlapped chunks that is ready for index or compute embedding
    @param body:
    @param shingle_size:
    @return:
    """
    if shingle_size < 1:
        raise Exception("shingle_size mush be greater than 0!")
    if shingle_size == 1:
        # TODO(Liang) @Ted take a look here. Why size must gt 2?
        # shingle_size is length of PAA's answer. Not answer extracted from web_pages.
        # I want to cover QnA: How many screen can I pin in zoom? -> 9
        shingle_size = 2

    chunks = split_long_body_text(body, int(shingle_size / 2))
    return make_shingles(chunks)


def make_shingles(chunks: List[str]) -> List[str]:
    """
    Make overlapped windows.
    """
    out = []
    if len(chunks) <= 1:
        return chunks
    prev = None
    for c in chunks:
        if prev:
            out.append('%s. %s' % (prev, c,))
        prev = c
    return out


RE_NORMALIZE_SPACE = re.compile(r'[ \t ]+')


def markdown_to_text(md):
    md = md or ''
    # force to remove image as a workaround for https://app.asana.com/0/0/1200419102209576/f
    md = MD_IMAGE_RE.sub(' ', md)
    # force to fix bold as a workaround for https://app.asana.com/0/0/1200419102209575/f
    md = md.replace('**', ' ')
    html = markdown.markdown(md)
    return RE_NORMALIZE_SPACE.sub(' ', ''.join(BeautifulSoup(html, features="lxml").findAll(text=True)))





_SENTENCE_RE = re.compile(r'[.?!] ')
_SENTENCE_MIN_LEN = 5


def md_line_to_sentence(line: str) -> List[str]:
    """
    break a line of Markdown text to sentences while maintaining punctuation
    @param line:
    @return:
    """
    line = line or ''
    sentence_candidates = _SENTENCE_RE.split(line)
    punctuations = _SENTENCE_RE.findall(line)
    num_punctuations = len(punctuations)
    out = []
    curr = []
    open_bracket_cnt = 0
    close_bracket_cnt = 0
    curr_str_len = 0
    for i, sen in enumerate(sentence_candidates):
        curr.append(sen)
        sen_len = len(sen.strip())
        curr_str_len += sen_len
        if i < num_punctuations:
            # maintain the original punctuation
            curr.append(punctuations[i])
        open_bracket_cnt += sen.count('[')
        close_bracket_cnt += sen.count(']')
        if open_bracket_cnt != close_bracket_cnt:
            # do not break insider  [] which is clickable text
            continue
        if curr_str_len < _SENTENCE_MIN_LEN:
            # don't break on super short fragments
            continue
        out.append(''.join(curr))
        curr = []
        open_bracket_cnt = 0
        close_bracket_cnt = 0
        curr_str_len = 0
    if curr:
        out.append(''.join(curr))

    return out


MD_HEADER_RE = re.compile(r'#+\W+\w+')
MD_ORDERED_LIST_RE = re.compile(r'[\d]{1,2}\.($|\W+)')
MD_UNORDERED_LIST_RE = re.compile(r'[*\-+]($|\W+)')





MD_LINK_RE = re.compile(r'([^!]|^)\[[^\]]*\]\(.+\)')


def with_hyperlink(md_line: str) -> bool:
    return bool(MD_LINK_RE.search(md_line or ''))


MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\(.+\)')


def with_image(md_line: str) -> bool:
    return bool(MD_IMAGE_RE.search(md_line or ''))


def with_video(md_line: str) -> bool:
    md_line = md_line or ''
    if MD_LINK_RE.search(md_line):
        # TODO: This is very hacky.
        return 'youtube.com' in md_line
    return False




























_CONTRACTION_RE = [(re.compile(regex), replace) for (regex, replace) in
                   _CONTRACTION_PATTERNS]


def expand_contraction(text: str) -> str:
    """
    Expansion of abbreviations, for example, "I'm" -> "I am"
    @param text:
    @return:
    """
    text = text.replace("’", "'")
    for (pattern, replace) in _CONTRACTION_RE:
        (text, count) = re.subn(pattern, replace, text)
    return text


nlp = get_spacy_lite()
stemmer = SnowballStemmer('english')
wn_lemma = WordNetLemmatizer()

# preload lemmatize to memory
wn_lemma.lemmatize('')


def pipeline(string_input, funcs):
    return reduce(lambda x, y: y(x), funcs, string_input)


def query_normalization(text, to_filter_stopword=False,
                        stop_words=_STOP_WORDS,
                        to_stem=False,
                        to_lemma=False) -> str:
    """
    8 steps:
    1. normalize whitespace:
        1) only use space character,
        2) clean up other whitespace characters: e.g.\s: \f\n\r\t\v,
            unicode: english \u0020, chinese \u3000, office \u00A0
        3) remove question mark at the end
        4) collapse multiple consecutive spaces into one
    2. spacial head norm.  e.g. Q:, q:, **, '\\b'
    3. expand_contraction. e.g. what's -> what is
    4. unicode normalization. e.g. \\u559c -> '喜'
    5. token_norm with choice of removing stop words.
        e.g. h.323/sip, keep the prase, do not split to h 323 / sip.
    6. token_lemma keep the orignal format. e.g. meetings -> meeting
    7. token_stem with retain the special words. e.g. ID -> id, zooms-> zoom,
        ios -> ios
    @param text:
    @param to_filter_stopword:
    @param stop_words:
    @param to_stem:
    @param to_lemma:
    @return:
    """
    text = pipeline(text, [whitespace_normalization,
                           head_normalization,
                           expand_contraction,
                           unicode_normalization])

    normalized_tokens = token_normalization(text, nlp,
                                            to_filter_stopword=to_filter_stopword,
                                            stop_words=stop_words)
    normalized_tokens = token_stemming(normalized_tokens, to_stem=to_stem)
    normalized_tokens = token_lemmatization(normalized_tokens, to_lemma=to_lemma)

    return ' '.join(normalized_tokens)


def whitespace_normalization(text):
    return " ".join(_SPACE_END_MARK_RE.sub(" ", text).split())


def head_normalization(text):
    text.replace('\\b', ' ')
    text = _Q_TAIL_RE.sub('', text)
    return _Q_HEAD_RE.sub("", text).strip()


def unicode_normalization(text):
    return _UNICODE_RE.sub(
        lambda x: x.group(1).encode("utf-8").decode("unicode-escape"), text)


def token_normalization(text, nlp,
                        to_filter_stopword: bool = False,
                        stop_words=_STOP_WORDS
                        ) -> List[str]:
    """
    note:
        1. use Doc(nlp.vocab, words=words)  not nlp(text)
          to retain the customer phrases.
        2. when use token.is_stop, e.g. 'what is zoom' -> 'zoom',
           'how much is zoom room' -> 'zoom room'.
           there are 326 stop words in spacy, but most of the word is useful
           in zoom, so we need to customize our own stop words list.
    @param nlp:
    @param to_filter_stopword:
    @param stop_words:
    @param text:
    @return:
    """
    words = text.split() or ''
    nlp_tokens = Doc(nlp.vocab, words=words)
    normalized_tokens = []
    for token in nlp_tokens:
        if to_filter_stopword and str.lower(token.text) in stop_words:
            continue
        normalized_tokens.append(token.norm_.lower())
    return normalized_tokens


def token_stemming(normalized_tokens, to_stem):
    """
    spaCy token to normalized text str
    alternative:
        porter_stemmer = PorterStemmer()
        wn_lemma = WordNetLemmatizer()
    """
    if to_stem:
        stem = []
        for token in normalized_tokens:
            if token.lower() in _SPECIAL_ABBREVIATIONS:
                stem.append(token.lower())
            else:
                stem.append(stemmer.stem(token))
        return stem
    return normalized_tokens


def token_lemmatization(normalized_tokens: List[str], to_lemma=False):
    if to_lemma:
        lemma = []
        for token in normalized_tokens:
            if token.lower() in _SPECIAL_ABBREVIATIONS:
                lemma.append(token.lower())
            else:
                lemma.append(wn_lemma.lemmatize(token))
        return lemma
    return normalized_tokens


def remove_segs_from_query(query: str, segs_to_remove: List[str]) -> str:
    new_query = copy.copy(query)
    for s in segs_to_remove:
        if s not in new_query:
            logger.warning(f"Can't remove: {s} not in {new_query}")
            continue
        idx = new_query.index(s)
        new_query = new_query[:idx] + new_query[idx + len(s):]
    return new_query.strip()

