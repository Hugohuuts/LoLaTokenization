import re
import spacy
NLP = spacy.load("en_core_web_sm")


import polars as pl

# Read Excel file directly into a Polars DataFrame
dfs  = pl.read_excel("MorphoLEX_en.xlsx", sheet_id=range(2,30), columns=["Word", "MorphoLexSegm"])
df = pl.concat(dfs.values())

def strip_entry(entry):
    """
    Converts a morphological entry like {(zoo)(log)>ic>>al>} 
    by removing all the unneeded symbols. Called by get_morphological_components
    
    Args:
        entry: The morphological entry as a string.
    
    Returns:
        A string of the tokenized word.
    """
    # Remove outer curly braces
    entry = entry.strip("{}")
    
    # Match components  to find every case of special delimiters like '>' or ')>'
    # Pattern created by ChatGPT
    pattern = r"\(([^)]+)\)|>([^>]+)"
    matches = re.findall(pattern, entry)
    
    # Flatten the matches and filter out empty components
    components = [match[0] or match[1] for match in matches]
    
    return components

def get_morphological_components(word, separator):
    """
    Splits a word into its morphological components.
    ---
    Args:
        word: A given word in string format
        data: a dataset that contains the morphological split for every word (in this case an excel file). 
    Returns:
        A string of the tokenized word.
    """
    # Search for the word in the combined DataFrame
    filtered = df.filter(df["Word"] == word)
    item = filtered["MorphoLexSegm"].first() if not filtered.is_empty() else None
    if item:
        # Return the morphemes as a list
        # Convert the morphemes to a string in the correct format
        tokenized = strip_entry(item)
        tokenized = f' {separator}'.join(tokenized).split()
        return tokenized
    else:
        # If the word is unknown, return it in full
        return [word]

def pos_tokenizer(sentence, tags, separator, special_marker):
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
         split = get_morphological_components(token.text, separator)
         split[0] = f"{special_marker}{split[0]}"
         tokenized.extend(split)
      else:
        tokenized.append(f"{special_marker}{token.text}")
    return tokenized

def noun_tokenizer(premises_hypothesis, separator, space_marker):
   return pos_tokenizer(premises_hypothesis[0], ['NOUN', 'PROPN'], separator, space_marker), pos_tokenizer(premises_hypothesis[1], ['NOUN', 'PROPN'], separator, space_marker),

def verb_tokenizer(premises_hypothesis, separator, space_marker):
   return pos_tokenizer(premises_hypothesis[0], ['VERB'], separator, space_marker), pos_tokenizer(premises_hypothesis[1], ['VERB'], separator, space_marker)

def adjective_tokenizer(premises_hypothesis, separator, space_marker):
   return pos_tokenizer(premises_hypothesis[0], ['ADJ'], separator, space_marker), pos_tokenizer(premises_hypothesis[1], ['ADJ'], separator, space_marker)  

#print(noun_tokenizer(["this contains categories and phrases aswell","mistakes intricaties overcompensation"], '#', '$'))

#print(get_morphological_components('zoological', '#'))
#print(get_morphological_components('jhchdjjdj', '#'))