# Tokenization Comparison Project

This project compares different tokenization algorithms (BPE, WordPiece, and SentencePiece) and demonstrates mask prediction using the RoBERTa model.

## Project Structure

- `tokenise.py`: Main Python script that performs tokenization and mask prediction
- `predictions.json`: JSON file containing tokenization results and mask predictions
- `compare.md`: Analysis of the differences between tokenization algorithms
- `.env`: Environment file for storing the Hugging Face API key (not included in repository)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install transformers tokenizers sentencepiece python-dotenv
```

3. Create a `.env` file in the project root with your Hugging Face API key:

```
HUGGINGFACE_API_KEY=your_api_key_here
```

## Usage

Run the tokenization and prediction script:

```bash
python tokenise.py
```

This will:
1. Tokenize the example sentence using three different algorithms
2. Perform mask prediction using RoBERTa
3. Save the results to `predictions.json`
4. Display the results in the console

## Results

The script analyzes the sentence: "The cat sat on the mat because it was tired."

It provides:
- Token lists and IDs for each tokenization algorithm
- Token counts for comparison
- Mask predictions for two masked tokens
- Plausibility analysis of the predictions

For a detailed analysis of the tokenization differences, see `compare.md`. 