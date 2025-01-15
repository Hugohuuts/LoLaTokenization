import spacy
from nltk.corpus import words
import nltk
NLP = spacy.load("en_core_web_sm")
nltk.download('words')
english_words = set(words.words())

def split_word_hybrid(word):
    """
    Function to tokenize a single word from a sentence. Called by pos_tokenizer
    ---
    Args:
        word: The word in string format
    Returns:
        A string tokenized by adding spaces between components
    """
    # Step 1: Try dictionary-based splitting
    components = ""
    for i in range(1, len(word)):
        prefix = word[:i]
        suffix = word[i:]
        if prefix in english_words and suffix in english_words:
            components += f"{prefix} {suffix}"
            break
    else:
        # Step 2: Fallback to subword splitting (e.g., character bigrams)
        components += ' '.join(word)
    
    return components


def pos_tokenizer(sentence, tags, separator):
    """
    Custom tokenizer that only tokenizes words that possess a specific POS tag
    ---
    Args:
        sentence: A given sentence in string format
        tags: a list of string that describe which tasks should be tokenized (for nouns, verbs or adjectives) 
        separator: special character(s) use by tokenizers when splitting words (e.g., `##` for BERT).
    Returns:
        A string of the tokenized sentence
    """
    tokenized = []
    for token in NLP(sentence):
      if token.pos_ in tags:
        tokenized.append(f"{separator}{split_word_hybrid(token.text)}")
      elif token.pos_ == "PUNCT":
        tokenized.append(token.text)
      else:
        tokenized.append(f"{separator}{token.text}")
    return ' '.join(tokenized)

def noun_tokenizer(premises_hypothesis, separator):
   return pos_tokenizer(premises_hypothesis[0], ['NOUN', 'PROPN'], separator), pos_tokenizer(premises_hypothesis[1], ['NOUN', 'PROPN'], separator),

def verb_tokenizer(premises_hypothesis, separator):
   return pos_tokenizer(premises_hypothesis[0], ['VERB'], separator), pos_tokenizer(premises_hypothesis[1], ['VERB'], separator)

def adjective_tokenizer(premises_hypothesis, separator):
   return pos_tokenizer(premises_hypothesis[0], ['ADJ'], separator), pos_tokenizer(premises_hypothesis[1], ['ADJ'], separator)  


premises_hypothesis = ("The cat is large. An ant is small.", "The cat is bigger than the ant.")
print(noun_tokenizer(premises_hypothesis, "#"))
print(verb_tokenizer(premises_hypothesis, "#"))
print(adjective_tokenizer(premises_hypothesis, "#"))

