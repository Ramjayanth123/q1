import os
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
import torch

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    print("Warning: HUGGINGFACE_API_KEY not found in .env file")

# Set Hugging Face API token
os.environ["HUGGINGFACE_TOKEN"] = api_key

# Example sentence
sentence = "The cat sat on the mat because it was tired."
print(f"Original sentence: {sentence}\n")

# Function to format tokens and IDs
def format_tokens_and_ids(tokens, ids):
    return {
        "tokens": tokens,
        "token_ids": ids,
        "token_count": len(tokens)
    }

# 1. BPE Tokenization (GPT-2)
print("Performing BPE tokenization...")
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bpe_tokens = bpe_tokenizer.tokenize(sentence)
bpe_ids = bpe_tokenizer.encode(sentence)
bpe_results = format_tokens_and_ids(bpe_tokens, bpe_ids)

print(f"BPE Tokens: {bpe_tokens}")
print(f"BPE IDs: {bpe_ids}")
print(f"BPE Token Count: {len(bpe_tokens)}\n")

# 2. WordPiece Tokenization (BERT)
print("Performing WordPiece tokenization...")
wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
wordpiece_tokens = wordpiece_tokenizer.tokenize(sentence)
wordpiece_ids = wordpiece_tokenizer.encode(sentence)
wordpiece_results = format_tokens_and_ids(wordpiece_tokens, wordpiece_ids)

print(f"WordPiece Tokens: {wordpiece_tokens}")
print(f"WordPiece IDs: {wordpiece_ids}")
print(f"WordPiece Token Count: {len(wordpiece_tokens)}\n")

# 3. SentencePiece (Unigram) Tokenization (T5)
print("Performing SentencePiece (Unigram) tokenization...")
sp_tokenizer = AutoTokenizer.from_pretrained("t5-small")
sp_tokens = sp_tokenizer.tokenize(sentence)
sp_ids = sp_tokenizer.encode(sentence)
sp_results = format_tokens_and_ids(sp_tokens, sp_ids)

print(f"SentencePiece Tokens: {sp_tokens}")
print(f"SentencePiece IDs: {sp_ids}")
print(f"SentencePiece Token Count: {len(sp_tokens)}\n")

# 4. Mask and Predict with RoBERTa
print("Performing mask prediction with RoBERTa...")
model_name = "roberta-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create masked sentences - mask "the" and "it"
masked_sentence_1 = sentence.replace("the mat", f"{roberta_tokenizer.mask_token} mat")
masked_sentence_2 = sentence.replace("it was", f"{roberta_tokenizer.mask_token} was")

fill_mask_pipeline = pipeline("fill-mask", model=model_name)

# Get predictions for first masked token
predictions_1 = fill_mask_pipeline(masked_sentence_1, top_k=3)
print(f"\nPredictions for '{masked_sentence_1}':")
for pred in predictions_1:
    print(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}, Sequence: {pred['sequence']}")

# Get predictions for second masked token
predictions_2 = fill_mask_pipeline(masked_sentence_2, top_k=3)
print(f"\nPredictions for '{masked_sentence_2}':")
for pred in predictions_2:
    print(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}, Sequence: {pred['sequence']}")

# Prepare results for JSON output
results = {
    "original_sentence": sentence,
    "tokenization": {
        "bpe": bpe_results,
        "wordpiece": wordpiece_results,
        "sentencepiece": sp_results
    },
    "mask_predictions": {
        "masked_sentence_1": {
            "sentence": masked_sentence_1,
            "predictions": [
                {
                    "token": pred["token_str"],
                    "score": pred["score"],
                    "sequence": pred["sequence"]
                } for pred in predictions_1
            ]
        },
        "masked_sentence_2": {
            "sentence": masked_sentence_2,
            "predictions": [
                {
                    "token": pred["token_str"],
                    "score": pred["score"],
                    "sequence": pred["sequence"]
                } for pred in predictions_2
            ]
        }
    }
}

# Save results to JSON file
with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to predictions.json") 