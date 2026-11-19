# Week 4 Tokenizer Lab

Compare pretrained BPE, WordPiece, and SentencePiece tokenizers on English,
Telugu, and Hindi text. The script prints each tokenizer's token pieces,
token IDs, and token count.

## Run

From this folder, install the dependencies and run the comparison:

```bash
python -m pip install -r requirements.txt
python tokenizer_comparison.py
```

The first run needs internet access to download the pretrained tokenizer files.
This lab compares pretrained models with different vocabularies and training
data, so token counts reflect both the tokenizer and its model training.
