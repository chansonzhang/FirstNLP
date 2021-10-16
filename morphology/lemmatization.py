from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB

lemmatizer = WordNetLemmatizer()
sent = ['was', 'invited']
lem_sent = [lemmatizer.lemmatize(words_sent, pos=VERB) for words_sent in sent]
print(lem_sent)
