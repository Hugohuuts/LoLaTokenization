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


def pos_tokenizer(sent, tags):
    """
    Tokenizer that splits words based on their respective POS tags
    """
    s = []
    for token in NLP(sent):
      if token.pos_ in tags:
        components = split_noun_hybrid(token.text)
        s.extend(components)
        #for char in token.text:
          #if char not in encoding.special_tokens:
            #s.append(char)
      else:
        s.append(token.text)
    return ' '.join(s)

sent = "This is a sample sentence that includes excrutiatingly lengthy components and rainfall can be divided into smaller parts."
print(pos_tokenizer(sent, ['NOUN', 'PROPN']))

