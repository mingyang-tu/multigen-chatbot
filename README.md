# ADL Final Project - Topic Transition Through Dialogue

## Introduction
This is a PyTorch implementation of a chatbot that can smoothly guide the conversation to some specific topics. 

### Model
We use [Multigen](https://github.com/cdjhz/multigen) as our generation model. 

### Dataset
The dataset is generated from [facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill) by ourselves.

| Train |  Dev  |
| ----- | ----- |
| 17683 |  1965 |

## Requirements
- spacy==3.3.0
- torch==1.11.0
- transformers==4.19.0
- fairseq==0.10.0
- nltk==3.4.5
- networkx==2.1
- datasets
- torch-scatter

## Usage

Download required files.

```
wget https://www.dropbox.com/s/ckovil7pe1gaz23/data.zip?dl=1 -O data.zip
unzip data.zip
```

### Inference
```
python -W ignore \
simulator.py \
--split test \
--num_chats 980 \
--model_name_or_path ./gpt2-dialog
```

### Display results
#### Format chats
```
python formatDialogue.py -f output.jsonl
```
#### Display
```
cat output-formatted.txt
```

### Evaluate
The evaluation metric we use is ```hit rate```, which is the percentage of conversations that contain keywords in all conversations.

```
python hit.py --prediction output.jsonl
```

## Citation

```
@inproceedings{ji2020language,
    title = "Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph",
    author = "Ji, Haozhe and Ke, Pei and Huang, Shaohan and Wei, Furu and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    year = "2020",
}
```