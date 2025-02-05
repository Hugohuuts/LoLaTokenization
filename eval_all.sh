tok_methods_pos=("adj" "char" "noun" "verb")
tok_methods_length=("unigram_tokenizer")
tok_methods_greedy=("greedy_prefix_tokenization" "greedy_suffix_tokenization" "greedy_longest_tokenization")
tok_methods_adverserial=("adverserial_shuffle_letters" "adverserial_pos_synonym_noun")
tok_methods_all=("adj" "char" "noun" "verb" "unigram_tokenizer" "greedy_prefix_tokenization" "greedy_suffix_tokenization" "greedy_longest_tokenization" "adverserial_shuffle_letters" "adverserial_pos_synonym_noun")

data_set_paths=(
    "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
    # "data/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    "data/snli_1.0/snli_1.0_test.jsonl"
)
# or if locally
# data_set_paths=(
#     "multinli_1.0_dev_mismatched.jsonl"
#     "multinli_1.0_dev_matched.jsonl"
#     "snli_1.0_test.jsonl"
# )

# methods_to_loop=${tok_methods_pos[@]}
# methods_to_loop=${tok_methods_greedy[@]}
# methods_to_loop=${tok_methods_length[@]}
# methods_to_loop=${tok_methods_adverserial[@]}
methods_to_loop=${tok_methods_all[@]}

models=("roberta" "minilm" "bart")
for model in ${models[@]}; do
    maximum_threads=3
    if [ $model == "bart" ]; then # BART takes much more memory
        maximum_threads=1
    fi
    for dataset in ${data_set_paths[@]}; do
        thread_limit=0
        for tok in ${methods_to_loop[@]}; do
            python eval_scripts.py --path $dataset --tok_method $tok --model $model --do_custom &
            thread_limit=$((thread_limit+1))
            if ((thread_limit > 0)) && ((thread_limit % maximum_threads == 0)); then
                wait
            fi
        done
        wait

        python eval_scripts.py --path $dataset --model $model
    done
done
