# Tokenization & Inference in NLI: A Comparative Study of Methods and Models

## Project Overview

This repository is part of a group project for the Logic and Language course at Utrecht Univerity, taught by Dr. Lasha Abzianidze PhD.
We will be exploring the impact of different tokenization techniques on Natural Language Inference (NLI), using three pre-trained models like BART, DistilRoBERTa, and MiniLM2.

## Objectives

- Investigate how different tokenization strategies affect NLI model performance.
- Use the SNLI and MNLI datasets for testing and comparison.

## Repository Structure

* `plots` - bar plots to help visualize the differences in model performance.
* `results` - raw `jsonl` result files containing the predicted label, accuracy, and probabilities related to the prediction process. Each used model has its own folder, where each tokenization approach we defined in our project is used for that model.
* `shap_analysis` - code to evaluate the feature importance of tokens from DistilRoBERTa and MiniLM2 on SNLI.
* `tokenization_methods` - main module containg all of the different tokenization approaches.
* `undertrained` - contains the reports and results generated by the code of Sander Land and Max Bartolo. 2024. [Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models](https://aclanthology.org/2024.emnlp-main.649/)
* Other:
    * `eval_*.ipynb` - notebooks to evaluate the models' performance 
    * `eval_all.sh` - run all experiments concurrently
    * `custom_*.py` - custom python classes for tokenizers and models to interface with our customised tokenization methods.
    * `prediction_utilities.py` - methods to obtain labels from a given model.
    * `tokenization_verification.py` - verify that no errors occur durring tokenization and that the resulting tokens fit within the context window of a mdoel.

## Environment Setup

req file

## Running the code

## Results and Analysis

### Table 1: Results for each model and tokenization combination, across all datasets

We report the accuracy and distribution of predicted labels (formatted as such: contradiction/entailment/neutral).  
Each table section presents results for a different model: **DistilRoBERTa**, **BART-Large**, and **MiniLM2**.  
Each table is divided into sections according to the tokenization approach.  
MNLI-M refers to the matched development set, and MNLI-MM refers to the mismatched development set.

| Tokenization | SNLI (Acc, Label %) | MNLI-M (Acc, Label %) | MNLI-MM (Acc, Label %) |
|-------------|---------------------|----------------------|----------------------|
| **DistilRoBERTa** |  |  |  |
| Native | 0.86 (24/50/26%) | 0.76 (27/41/32%) | 0.79 (22/52/26%) |
| Character | 0.34 (0/99/1%) | 0.36 (1/97/2%) | 0.36 (1/97/2%) |
| Noun | 0.87 (33/34/33%) | 0.80 (34/34/32%) | 0.81 (33/35/32%) |
| Verb | 0.87 (33/35/32%) | 0.80 (33/34/33%) | 0.81 (33/35/32%) |
| Adjective | 0.87 (33/34/33%) | 0.80 (33/34/33%) | 0.80 (33/35/32%) |
| Greedy Suffix | 0.37 (5/94/1%) | 0.38 (5/91/4%) | 0.38 (5/92/3%) |
| Greedy Prefix | 0.37 (4/94/2%) | 0.40 (5/91/4%) | 0.39 (5/91/4%) |
| Unigram | **0.88 (33/34/33%)** | **0.82 (33/34/33%)** | **0.82 (33/35/32%)** |
| ALS | 0.76 (35/30/35%) | 0.69 (33/29/38%) | 0.70 (32/30/38%) |
| ANSS | 0.80 (34/33/33%) | 0.78 (34/31/35%) | 0.78 (33/33/34%) |

| Tokenization | SNLI (Acc, Label %) | MNLI-M (Acc, Label %) | MNLI-MM (Acc, Label %) |
|-------------|---------------------|----------------------|----------------------|
| **BART-Large** |  |  |  |
| Native | **0.90 (33/33/34%)** | 0.84 (31/30/39%) | 0.86 (30/32/38%) |
| Character | 0.46 (12/59/29%) | 0.48 (18/60/22%) | 0.50 (17/62/21%) |
| Noun | 0.86 (27/33/40%) | 0.86 (30/32/38%) | 0.86 (30/33/37%) |
| Verb | 0.86 (28/33/39%) | 0.85 (31/32/37%) | 0.85 (30/33/37%) |
| Adjective | 0.86 (28/33/39%) | 0.85 (30/32/38%) | 0.85 (30/32/38%) |
| Greedy Suffix | 0.46 (42/50/8%) | 0.46 (37/45/18%) | 0.46 (37/44/19%) |
| Greedy Prefix | 0.46 (40/52/8%) | 0.46 (31/52/17%) | 0.47 (32/51/17%) |
| Unigram | **0.90 (33/33/34%)** | **0.88 (32/33/35%)** | **0.88 (32/33/35%)** |
| ALS | 0.78 (34/27/39%) | 0.74 (33/25/42%) | 0.75 (33/26/41%) |
| ANSS | 0.83 (34/32/34%) | 0.83 (33/30/37%) | 0.84 (32/30/37%) |

| Tokenization | SNLI (Acc, Label %) | MNLI-M (Acc, Label %) | MNLI-MM (Acc, Label %) |
|-------------|---------------------|----------------------|----------------------|
| **MiniLM2** |  |  |  |
| Native | 0.88 (34/46/20%) | 0.80 (29/39/32%) | 0.82 (29/51/20%) |
| Character | 0.34 (3/97/0%) | 0.36 (7/92/1%) | 0.37 (6/93/1%) |
| Noun | 0.78 (31/36/33%) | 0.83 (34/33/33%) | 0.83 (33/34/33%) |
| Verb | 0.68 (35/44/21%) | 0.82 (34/33/33%) | 0.83 (33/34/33%) |
| Adjective | 0.66 (33/41/26%) | 0.82 (34/33/33%) | 0.83 (33/34/33%) |
| Greedy Suffix | 0.40 (57/42/1%) | 0.39 (44/55/1%) | 0.40 (42/57/1%) |
| Greedy Prefix | 0.44 (46/53/1%) | 0.40 (34/65/1%) | 0.41 (34/65/1%) |
| Unigram | **0.89 (33/34/33%)** | **0.84 (33/34/33%)** | **0.85 (33/35/32%)** |
| ALS | 0.76 (36/30/34%) | 0.71 (37/27/36%) | 0.72 (34/29/37%) |
| ANSS | 0.81 (34/32/34%) | 0.80 (34/31/35%) | 0.81 (33/32/35%) |

---

### Table 2: Modification rate for POS tokenizers

Modification rate refers to the percentage of words within the SNLI dataset that are successfully split into their morphological components and change as a result.  
Words included in MorphoLex but not changed by this transformation do not contribute to this percentage.

| POS Tokenizer | Modification Rate (%) |
|--------------|----------------------|
| Noun        | 8.7%  |
| Verb        | 10.3% |
| Adjective   | 1.6%  |

## Contact

We would love to get feedback from the community. If you have any questions, please open an issue or contact us.

This repository is developed by the following students from Utrecht University (UU) as part of the course Logic and Language:

- Yme Boland: [E-mail](mailto:y.m.dejong%40students.uu.nl?subject=UU%20LoLa%20Tokenization%20Project), [GitHub](https://github.com/6950167)
- Shane Siwpersad: [E-mail](mailto:s.r.d.siwpersad%40students.uu.nl?subject=UU%20LoLa%20Tokenization%20Project)
- Eduard Köntöş: [E-mail](mailto:e.r.kontos%40students.uu.nl?subject=UU%20LoLa%20Tokenization%20Project), [GitHub](https://github.com/Tron404)
- Hugo Voorheijen: [E-mail](mailto:h.j.a.voorheijen%40students.uu.nl?subject=UU%20LoLa%20Tokenization%20Project), [GitHub](https://github.com/Hugohuuts)
- Julius Bijkerk: [E-mail](mailto:j.j.bijkerk%40students.uu.nl?subject=UU%20LoLa%20Tokenization%20Project), [GitHub](https://github.com/JungCesar)

## Acknowledgements

First of all, we would like to thank Dr. Lasha Abzianidze PhD ([GitHub](https://github.com/kovvalsky), [Website](https://naturallogic.pro)), for his guidance and feedback throughout this project.
His expertise and insights have significantly shaped the direction and execution of our research.
Secondly, we are thankful for the students, participating in the course Logic and Language 24/25 at Utrecht University , for their presentations, broadening our knowledge on the topic of NLI.
Finally, we would like to recognize the strength of the open-source models and datasets, especially those used in our experiment, highlighting how collaborative efforts facilitate new research.

## References

We have used and built upon the work of others, the list below represents the most important sources that formed a basis of our work.

- **WordNet** — [wordnet.princeton.edu](https://wordnet.princeton.edu/). [Accessed 02-02-2025].

- **Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning.** 2015.  
  *A large annotated corpus for learning natural language inference.*  
  In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 632–642, Lisbon, Portugal.  
  Association for Computational Linguistics.

- **Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al.** 2024.  
  *A survey on evaluation of large language models.*  
  *ACM Transactions on Intelligent Systems and Technology*, 15(3):39:2.

- **Jeremy Howard and Sebastian Ruder.** 2018.  
  *Universal language model fine-tuning for text classification.*  
  arXiv preprint [arXiv:1801.06146](https://arxiv.org/abs/1801.06146).

- **Sander Land and Max Bartolo.** 2024.  
  *Fishing for magikarp: Automatically detecting under-trained tokens in large language models.*  
  In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 11631–11646, Miami, Florida, USA.  
  Association for Computational Linguistics.

- **Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer.** 2019.  
  *BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.*  
  CoRR, [abs/1910.13461](https://arxiv.org/abs/1910.13461).

- **Scott Lundberg.** 2017.  
  *A unified approach to interpreting model predictions.*  
  arXiv preprint [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).

- **Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela.** 2019.  
  *Adversarial NLI: A new benchmark for natural language understanding.*  
  arXiv preprint [arXiv:1910.14599](https://arxiv.org/abs/1910.14599).

- **Nils Reimers and Iryna Gurevych.** 2019.  
  *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.*  
  In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.*  
  Association for Computational Linguistics.

- **Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, and Wei Liu.** 2024.  
  *Token-modification adversarial attacks for natural language processing: A survey.*  
  *AI Communications*, 37(4):655–676.

- **Claudia H. Sánchez-Gutiérrez, Hugo Mailhot, S. Hélène Deacon, and Maximiliano A. Wilson.** 2018.  
  *MorphoLex: A derivational morphological database for 70,000 English words.*  
  *Behavior Research Methods*, 50:1568–1580.

- **Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.** 2019.  
  *DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter.*  
  CoRR, [abs/1910.01108](https://arxiv.org/abs/1910.01108).

- **Rico Sennrich, Barry Haddow, and Alexandra Birch.** 2015.  
  *Neural machine translation of rare words with subword units.*  
  CoRR, [abs/1508.07909](https://arxiv.org/abs/1508.07909).

- **Omri Uzan, Craig W. Schmidt, Chris Tanner, and Yuval Pinter.** 2024.  
  *Greed is all you need: An evaluation of tokenizer inference methods.*

- **Adina Williams, Nikita Nangia, and Samuel Bowman.** 2018.  
  *A broad-coverage challenge corpus for sentence understanding through inference.*  
  In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 1112–1122, New Orleans, Louisiana.  
  Association for Computational Linguistics.

- **Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean.** 2016.  
  *Google’s neural machine translation system: Bridging the gap between human and machine translation.*  
  CoRR, [abs/1609.08144](https://arxiv.org/abs/1609.08144).
