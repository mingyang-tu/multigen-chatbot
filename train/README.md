## Training

### Requirements

- torch==1.11.0
- transformers==2.8.0
- fairseq==0.10.0
- nltk==3.4.5
- networkx==2.1
- spacy==2.2.1
- torch-scatter

### Preprocessing

Download the pre-trained GPT-2 model.

```
mkdir -p models
cd models
mkdir -p gpt2-small
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json
```

Preprocessing multi-hop relational paths for the model.

```
python ground_concepts_simple.py dialog
python find_neighbours.py dialog
python filter_triple.py dialog
```

### Training
```
python3 main.py \
--train_data_file ../dialog_dataset/train \
--dev_data_file  ../dialog_dataset/dev \
--test_data_file ../dialog_dataset/dev \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ../models/gpt2-dialog \
--source_length 96 \
--target_length 32 \
--model_type gpt2 \
--model_name_or_path ../models/gpt2-small/ \
--do_train \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 10 \
--learning_rate 3e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--logging_steps 100 \
--save_last \
--validate_steps 1106
```
