import os
import json
import torch
from transformers import AutoTokenizer, pipeline

def main():
    # Get user input
    print("Enter a sentence to tokenize and analyze:")
    sentence = input().strip()
    
    if not sentence:
        sentence = "The cat sat on the mat because it was tired."
        print(f"No input provided. Using default sentence: {sentence}")
    
    print(f"\nAnalyzing sentence: {sentence}\n")

    # Function to format tokens and IDs
    def format_tokens_and_ids(tokens, ids):
        return {
            "tokens": tokens,
            "token_ids": ids,
            "token_count": len(tokens)
        }

    try:
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

        # Ask user which words to mask
        print("Enter two words from your sentence to mask (separated by comma):")
        mask_words = input().strip()
        
        if not mask_words or "," not in mask_words:
            print("No valid mask words provided. Using default masks.")
            # Find two words to mask (preferably nouns or pronouns)
            words = sentence.split()
            if len(words) >= 4:
                mask_word1 = words[1]  # Usually a noun
                mask_word2 = words[-2]  # Usually the last content word
            else:
                mask_word1 = words[0] if words else "the"
                mask_word2 = words[-1] if len(words) > 1 else "it"
        else:
            mask_word1, mask_word2 = [word.strip() for word in mask_words.split(",", 1)]

        # 4. Mask and Predict with RoBERTa
        print("\nPerforming mask prediction with RoBERTa...")
        model_name = "roberta-base"
        roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create masked sentences
        masked_sentence_1 = sentence.replace(mask_word1, f"{roberta_tokenizer.mask_token}", 1)
        masked_sentence_2 = sentence.replace(mask_word2, f"{roberta_tokenizer.mask_token}", 1)
        
        print(f"Masked sentences:")
        print(f"1. {masked_sentence_1}")
        print(f"2. {masked_sentence_2}")

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
                    "masked_word": mask_word1,
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
                    "masked_word": mask_word2,
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

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have an internet connection and the required models are accessible.")

if __name__ == "__main__":
    main()
