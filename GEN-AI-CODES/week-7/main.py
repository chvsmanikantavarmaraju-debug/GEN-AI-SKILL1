"""Inspect GPT-2 next-token probabilities for a few sample prompts.

Install dependencies with: python -m pip install -r requirements.txt
Run with: python main.py
The first run downloads the GPT-2 model files.
"""

import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
PROMPTS = [
    "The cat is",
    "Artificial intelligence is",
    "India is a",
    "Machine learning helps",
]
TOP_K = 5
OUTPUT_FILE = Path(__file__).resolve().parent / "week7_probability_results.csv"


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully.")

    results = []

    for prompt in PROMPTS:
        print("\n" + "=" * 60)
        print("PROMPT:", prompt)
        print("=" * 60)

        inputs = tokenizer(prompt, return_tensors="pt")
        print("Input token IDs:", inputs["input_ids"][0].tolist())

        with torch.inference_mode():
            outputs = model(**inputs)

        # The final sequence position predicts the next token.
        logits = outputs.logits[0, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, min(TOP_K, probabilities.numel()))

        print(f"\nTop {len(top_ids)} next-token candidates:")
        print("-" * 50)
        for rank, (token_id, probability) in enumerate(
            zip(top_ids.tolist(), top_probs.tolist()), start=1
        ):
            token = tokenizer.decode([token_id])
            print(
                f"Rank: {rank} | Token: {token!r} | "
                f"Token ID: {token_id} | Probability: {probability:.6f}"
            )
            results.append(
                [prompt, rank, token, token_id, round(probability, 6)]
            )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Prompt", "Rank", "Candidate Token", "Token ID", "Probability"]
        )
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
