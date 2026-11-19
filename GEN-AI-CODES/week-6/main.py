"""Run a first-pass golden-question evaluation against a local Ollama model."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
INPUT_FILE = BASE_DIR / "golden_set.json"
OUTPUT_FILE = BASE_DIR / "baseline_results.csv"
RELEASE = "v0.1"


def load_golden_questions():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("golden_set.json must contain a JSON list.")

    required = {"id", "question", "expected_answer", "keywords", "source"}
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Question {index} is missing fields: {', '.join(sorted(missing))}"
            )
        if not isinstance(item["keywords"], list):
            raise ValueError(f"Question {index} keywords must be a list.")

    return questions


def get_llm_answer(question):
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False,
        # Keep the baseline responses concise so the full set finishes in a
        # reasonable time on a local model.
        "options": {"temperature": 0, "num_predict": 160},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    answer = result.get("response", "").strip()
    if not answer:
        raise ValueError("Ollama returned an empty response.")
    return answer


def calculate_score(answer, expected_keywords):
    normalized_answer = answer.casefold()
    matched = [
        keyword for keyword in expected_keywords
        if keyword.casefold() in normalized_answer
    ]
    missing = [
        keyword for keyword in expected_keywords
        if keyword.casefold() not in normalized_answer
    ]
    score = (len(matched) / len(expected_keywords) * 100) if expected_keywords else 0
    return round(score, 2), matched, missing


def evaluate_question(item):
    try:
        answer = get_llm_answer(item["question"])
        score, matched, missing = calculate_score(answer, item["keywords"])
        status = "PASS" if score >= 60 else "REVIEW"
        print(f"{item['id']}: {status} ({score:.2f}%)", flush=True)
        return {
            "Question_ID": item["id"],
            "Question": item["question"],
            "Expected_Answer": item["expected_answer"],
            "LLM_Answer": answer,
            "Matched_Keywords": ", ".join(matched),
            "Missing_Keywords": ", ".join(missing),
            "Score": score,
            "Status": status,
            "Source": item["source"],
            "Model": MODEL_NAME,
            "Release": RELEASE,
        }
    except (requests.RequestException, ValueError) as error:
        print(f"{item['id']}: ERROR - {error}", flush=True)
        return {
            "Question_ID": item["id"],
            "Question": item["question"],
            "Expected_Answer": item["expected_answer"],
            "LLM_Answer": "",
            "Matched_Keywords": "",
            "Missing_Keywords": ", ".join(item["keywords"]),
            "Score": 0,
            "Status": "ERROR",
            "Source": item["source"],
            "Model": MODEL_NAME,
            "Release": RELEASE,
        }


def save_results(results):
    fields = [
        "Question_ID", "Question", "Expected_Answer", "LLM_Answer",
        "Matched_Keywords", "Missing_Keywords", "Score", "Status",
        "Source", "Model", "Release",
    ]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    questions = load_golden_questions()
    print("GOLDEN QUESTION BASELINE")
    print(f"Questions: {len(questions)} | Model: {MODEL_NAME} | Release: {RELEASE}")
    print("Checking Ollama at", OLLAMA_URL)

    # Create the CSV immediately, then checkpoint after each question so a
    # long local-model run still leaves useful results if it is interrupted.
    results = []
    save_results(results)
    for item in questions:
        results.append(evaluate_question(item))
        save_results(results)

    successful = [result for result in results if result["Status"] != "ERROR"]
    passed = sum(result["Status"] == "PASS" for result in results)
    reviewed = sum(result["Status"] == "REVIEW" for result in results)
    errors = sum(result["Status"] == "ERROR" for result in results)
    average = (
        sum(result["Score"] for result in successful) / len(successful)
        if successful else 0
    )

    print("\nBASELINE SUMMARY")
    print(f"Passed: {passed} | Review: {reviewed} | Errors: {errors}")
    print(f"Average keyword coverage: {average:.2f}%")
    print("Evaluation date:", datetime.now().astimezone().isoformat(timespec="seconds"))
    print("Results saved to:", OUTPUT_FILE)
    print(f"Release {RELEASE} baseline completed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Could not start evaluation: {error}") from error
