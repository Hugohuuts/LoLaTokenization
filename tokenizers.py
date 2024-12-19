import spacy
from nltk.corpus import words
import nltk
NLP = spacy.load("en_core_web_sm")
nltk.download('words')
english_words = set(words.words())

def split_noun_hybrid(noun):
    """
    Split nouns using an english lexicon
    """
    # Step 1: Try dictionary-based splitting
    components = []
    for i in range(1, len(noun)):
        prefix = noun[:i]
        suffix = noun[i:]
        if prefix in english_words and suffix in english_words:
            components.extend([prefix, suffix])
            break
    #else:
        # Step 2: Fallback to subword splitting (e.g., character bigrams)
        #components = list(noun)
    
    return components if components else [noun]


def pos_tokenizer(premises, hypothesis, tags):
    """
    Tokenizer that splits words based on their respective POS tags
    """
    p = []
    for token in NLP(premises):
      if token.pos_ in tags:
        components = split_noun_hybrid(token.text)
        p.extend(components)
        #for char in token.text:
          #if char not in encoding.special_tokens:
            #s.append(char)
      else:
        p.append(token.text)
    h = []
    for token in NLP(hypothesis):
      if token.pos_ in tags:
        components = split_noun_hybrid(token.text)
        h.extend(components)
        #for char in token.text:
          #if char not in encoding.special_tokens:
            #s.append(char)
      else:
        h.append(token.text)
    return ' '.join(p), ' '.join(h)

premises = "The cat is large. An ant is small."
hypothesis = "The cat is bigger than the ant."
print(pos_tokenizer(premises, hypothesis, ['NOUN', 'PROPN']))

