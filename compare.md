# Tokenization Algorithm Comparison

## Overview
This document compares three tokenization algorithms: BPE (Byte-Pair Encoding), WordPiece, and SentencePiece (Unigram) on the sentence:

> The cat sat on the mat because it was tired.

## Token Counts
- **BPE (GPT-2)**: Typically produces 9-11 tokens
- **WordPiece (BERT)**: Typically produces 10-12 tokens
- **SentencePiece (T5)**: Typically produces 9-10 tokens

## Why the Splits Differ

The tokenization algorithms differ in their approach to segmenting text:

**BPE (Byte-Pair Encoding)** starts with individual characters and iteratively merges the most frequent pairs to form new tokens. It treats spaces as special characters and often generates tokens that start with a space (e.g., " the"). BPE is greedy and deterministic, always selecting the longest possible token from its vocabulary.

**WordPiece** is similar to BPE but uses a different merging criterion based on likelihood rather than frequency. It marks subwords with '##' prefixes (except for the first subword of a word) and tends to split words into more semantically meaningful units. It typically handles common words as single tokens.

**SentencePiece (Unigram)** treats the text as a sequence of Unicode characters without any pre-tokenization. It uses a probabilistic model to find the optimal segmentation, which can sometimes result in different splits than the other methods. It handles spaces as normal characters, often incorporating them into tokens, and can generate tokens that cross word boundaries.

These differences result in varying token counts and boundaries, affecting how models process and understand text. The choice of tokenizer can impact model performance on different tasks, especially for out-of-vocabulary words and morphologically rich languages. 