import re


_CONTRACTION_PATTERNS = [
    (r'won\'t', 'will not'),
    (r'[Cc]an\'t', 'cannot'),
    (r'[Dd]on\'t', 'do not'),
    (r'[Dd]oesn\'t', 'does not'),
    (r'([Ii]t|[Tt]his|[TtWw]hat|[TtWw]here|[Ww]ho)\'s', '\\g<1> is'),
    (r'([Ii])\'m', '\\g<1> am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\\g<1> will'),
    (r'(\w+)n\'t', '\\g<1> not'),
    (r'(\w+)\'ve', '\\g<1> have'),
    (r'(\w+)\'re', '\\g<1> are'),
    (r'(\w+)\'d', '\\g<1> would'),
    (r'isnt\b', 'is not'),
    (r'dont\b', 'do not'),
    (r'[wW]hats\b', 'what is'),
]

_SPECIAL_ABBREVIATIONS = {'ios', 'does', 'windows', 'sms', 'as'}
_Q_HEAD_RE = re.compile(r'(Q:)|(q:)|\*+')
_Q_TAIL_RE = re.compile(r"\.$")
_SPACE_END_MARK_RE = re.compile(r'(\s+)|[\u0020\u3000\u00A0]|[?？“”,]')
_UNICODE_RE = re.compile(r'(\\u[a-zA-Z0-9]{4})')

_STOP_WORDS = {'a', 'an', 'my', 'him', 'her', 'he', 'hers', 'ours', 'you', 'i',
               'the'}

