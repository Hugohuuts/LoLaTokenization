import re
import pandas as pd
import spacy
NLP = spacy.load("en_core_web_sm")


# Load the MorphoLex excel file
file_path = "MorphoLEX_en.xlsx"  # Replace with your actual file path
sheets = pd.read_excel(file_path, sheet_name=None)

# The first sheet contains information about the dataset, which is not needed
sheet_names = list(sheets.keys())[1:]
filtered_sheets = {name: sheets[name] for name in sheet_names}

# Combine the remaining sheets into a single DataFrame
data = pd.concat(filtered_sheets.values(), ignore_index=True)

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
    
    return " ".join(components)

def get_morphological_components(word):
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
    row = data[data['Word'].str.lower() == word.lower()]
    
    if not row.empty:
        # Return the morphemes as a list
        morphemes = row.iloc[0]['MorphoLexSegm']
        # Convert the morphemes to a string in the correct format
        return strip_entry(morphemes)
    else:
        # If the word is unknown, return it in full
        return word

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
         tokenized.append(f"{separator}{get_morphological_components(token.text)}")
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