tok_methods_pos=("adj" "char" "noun" "verb")
tok_methods_length=("unigram_tokenizer" "custom_tokenization_word_length" "bigram_tokenizer" "trigram_tokenizer")
tok_methods_greedy=("greedy_prefix_tokenization" "greedy_suffix_tokenization" "greedy_longest_tokenization")

data_set_paths=(
    "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
    "data/snli_1.0/snli_1.0_test.jsonl"
)

# methods_to_loop=${tok_methods_pos[@]}
methods_to_loop=${tok_methods_greedy[@]}
# methods_to_loop=${tok_methods_length[@]}
model=roberta
for dataset in ${data_set_paths[@]}; do
    thread_limit=0
    for tok in ${methods_to_loop[@]}; do
        python eval_scripts.py --path $dataset --tok_method $tok --model $model --do_custom &
        thread_limit=$((thread_limit+1))
        if ((thread_limit > 0)) && ((thread_limit % 2 == 0)); then
            wait
        fi
    done
    wait

#    python eval_scripts.py --path $dataset --model $model 
done
