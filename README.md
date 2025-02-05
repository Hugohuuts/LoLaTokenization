# Custom Tokenization and NLI Performance

By:

- Yme Boland
- Shane Siwpersad
- Eduard Köntöş
- Hugo Voorheijen
- Julius Bijkerk

## Project Overview

This repository is part of a group project for the Logic and Language course at Utrecht Univerity, taught by Dr. Lasha Abzianidze PhD.
We will be exploring the impact of different tokenization techniques on Natural Language Inference (NLI), using pre-trained models like BERT, RoBERTa, and other (foundational) models.

## Objectives

- Investigate how different tokenization strategies affect NLI model performance.
- Use the SNLI and MNLI datasets for testing and comparison.

## Repository Structure

Main folder: LoLaTokenization

## Getting Started

The results can be presented visually through methods within the eval_accuracy Notebook file.
To run an experiment, use eval_scripts.py with the desired parameters as such: 
python eval_scripts.py --path "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl" --tok_method adj --model roberta --do_custom true

## Contributions

Work on assigned tasks in personal branches.
Push updates regularly for review in weekly meetings.

