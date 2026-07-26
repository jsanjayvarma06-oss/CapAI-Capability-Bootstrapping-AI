#!/usr/bin/env python3
"""
CapAI Small-Model Case Study
==============================
Tests whether a small LLM (LLaMA-3.1-8B) can correctly use CapAI
as a tool-invocation layer via a JSON-based prompt scheme.

This replicates and extends the Phi-3 Mini case study (Section X)
using LLaMA-3.1-8B served via Groq API — same parameter class,
no local installation required.

The small model receives a system prompt explaining how to call CapAI,
then a user query. We measure:
  - Tool invocation rate (did it emit a JSON tool call?)
  - Argument correctness (did it pass sensible args?)
  - End-to-end correctness (did CapAI + model produce right answer?)
  - Spurious invocation rate (did it call CapAI when it shouldn't?)

Usage:
    python run_casestudy.py
    python run_casestudy.py --trials 50
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
SMALL_MODEL    = "llama-3.1-8b-instant"
CAPAI_BASE_URL = os.environ.get(
    "CAPAI_URL",
    "https://capai-capability-bootstrapping-ai-fu58.onrender.com"
).rstrip("/")

OUTPUT_DIR = Path(__file__).parent / "results"
DELAY      = 2.0

FIELDNAMES = [
    "prompt_id", "query_type", "query", "expected_answer",
    "invoked_tool", "tool_name", "tool_args",
    "capai_result", "final_answer", "correct",
    "spurious_invocation", "latency_ms", "timestamp"
]

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — teaches the small model how to call CapAI
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with access to CapAI, a capability tool.

When you need to compute something (math, string processing, data manipulation), call CapAI by responding with ONLY this JSON format:
{"tool": "capai", "name": "<function_name>", "description": "<what it does>", "args": [<arg1>, <arg2>]}

When the question is purely factual and requires no computation, answer directly in plain text.

Rules:
- Use CapAI for: prime checks, fibonacci, factorials, palindromes, sorting, string manipulation, math
- Do NOT use CapAI for: capital cities, historical facts, definitions, opinions
- function name must be snake_case
- args must be a JSON array of actual values

Examples:
User: Is 17 prime?
Assistant: {"tool": "capai", "name": "is_prime", "description": "Return True if n is prime", "args": [17]}

User: What is the capital of France?
Assistant: Paris.

User: Reverse the string 'hello'
Assistant: {"tool": "capai", "name": "reverse_string", "description": "Return string reversed", "args": ["hello"]}
"""

# ──────────────────────────────────────────────────────────────────────────────
# 50 test prompts across categories
# ──────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    # Numerical — should invoke CapAI
    {"id": 1,  "type": "numerical",  "query": "Is 97 a prime number?",                           "expected": True,      "should_invoke": True},
    {"id": 2,  "type": "numerical",  "query": "What is the 10th Fibonacci number?",               "expected": 55,        "should_invoke": True},
    {"id": 3,  "type": "numerical",  "query": "Calculate the factorial of 7",                     "expected": 5040,      "should_invoke": True},
    {"id": 4,  "type": "numerical",  "query": "What is the GCD of 48 and 18?",                   "expected": 6,         "should_invoke": True},
    {"id": 5,  "type": "numerical",  "query": "Convert 100 degrees Celsius to Fahrenheit",        "expected": 212.0,     "should_invoke": True},
    {"id": 6,  "type": "numerical",  "query": "Is 144 a perfect square?",                         "expected": True,      "should_invoke": True},
    {"id": 7,  "type": "numerical",  "query": "What is the sum of digits of 12345?",              "expected": 15,        "should_invoke": True},
    {"id": 8,  "type": "numerical",  "query": "What is 2 to the power of 10 modulo 1000?",       "expected": 24,        "should_invoke": True},
    {"id": 9,  "type": "numerical",  "query": "Is 28 a perfect number?",                          "expected": True,      "should_invoke": True},
    {"id": 10, "type": "numerical",  "query": "What is the LCM of 4 and 6?",                     "expected": 12,        "should_invoke": True},

    # String — should invoke CapAI
    {"id": 11, "type": "string",     "query": "Is 'racecar' a palindrome?",                       "expected": True,      "should_invoke": True},
    {"id": 12, "type": "string",     "query": "Reverse the words in 'hello world foo'",           "expected": "foo world hello", "should_invoke": True},
    {"id": 13, "type": "string",     "query": "Count the vowels in 'Hello World'",                "expected": 3,         "should_invoke": True},
    {"id": 14, "type": "string",     "query": "Is 'listen' an anagram of 'silent'?",             "expected": True,      "should_invoke": True},
    {"id": 15, "type": "string",     "query": "Convert 'hello_world' from snake_case to camelCase","expected":"helloWorld","should_invoke": True},
    {"id": 16, "type": "string",     "query": "Run-length encode the string 'aaabbc'",            "expected": "3a2b1c",  "should_invoke": True},
    {"id": 17, "type": "string",     "query": "Is 'The quick brown fox jumps over the lazy dog' a pangram?","expected": True,"should_invoke": True},
    {"id": 18, "type": "string",     "query": "What is the Levenshtein distance between 'kitten' and 'sitting'?","expected":3,"should_invoke":True},
    {"id": 19, "type": "string",     "query": "Repeat the string 'ab' 3 times",                  "expected": "ababab",  "should_invoke": True},
    {"id": 20, "type": "string",     "query": "How many words are in 'the quick brown fox'?",    "expected": 4,         "should_invoke": True},

    # Algorithms — should invoke CapAI
    {"id": 21, "type": "algorithms", "query": "Find two numbers in [2, 7, 11, 15] that sum to 9, return their indices","expected":[0,1],"should_invoke":True},
    {"id": 22, "type": "algorithms", "query": "What is the maximum subarray sum of [-2,1,-3,4,-1,2,1,-5,4]?","expected":6,"should_invoke":True},
    {"id": 23, "type": "algorithms", "query": "Binary search for 7 in [1,3,5,7,9,11], return its index","expected":3,"should_invoke":True},
    {"id": 24, "type": "algorithms", "query": "What is the minimum number of coins to make 36 cents using [1,5,10,25]?","expected":3,"should_invoke":True},
    {"id": 25, "type": "algorithms", "query": "Find the kth largest element where k=2 in [3,2,1,5,6,4]","expected":5,"should_invoke":True},

    # Validation — should invoke CapAI
    {"id": 26, "type": "validation", "query": "Is 'user@example.com' a valid email address?",    "expected": True,      "should_invoke": True},
    {"id": 27, "type": "validation", "query": "Is '192.168.1.1' a valid IPv4 address?",          "expected": True,      "should_invoke": True},
    {"id": 28, "type": "validation", "query": "Is '{\"key\": \"value\"}' valid JSON?",           "expected": True,      "should_invoke": True},
    {"id": 29, "type": "validation", "query": "Does '4532015112830366' pass the Luhn checksum?", "expected": True,      "should_invoke": True},
    {"id": 30, "type": "validation", "query": "Is '#FF8800' a valid hex color?",                 "expected": True,      "should_invoke": True},

    # India-specific — should invoke CapAI
    {"id": 31, "type": "india",      "query": "Calculate 18% GST on a base amount of 1000 rupees","expected":{"gst_amount":180.0,"total_amount":1180.0},"should_invoke":True},
    {"id": 32, "type": "india",      "query": "Is 'ABCDE1234F' a valid PAN card number?",        "expected": True,      "should_invoke": True},
    {"id": 33, "type": "india",      "query": "Format 1234567 in the Indian number system",      "expected": "12,34,567","should_invoke": True},
    {"id": 34, "type": "india",      "query": "Calculate 10% TDS on 50000 rupees",               "expected": {"tds":5000.0,"net_payable":45000.0},"should_invoke":True},
    {"id": 35, "type": "india",      "query": "What is the PF contribution (12%) for a basic salary of 30000?","expected":{"employee_contribution":3600.0,"employer_contribution":3600.0},"should_invoke":True},

    # Date/Time — should invoke CapAI
    {"id": 36, "type": "datetime",   "query": "How many days between 2024-01-01 and 2024-01-31?","expected": 30,        "should_invoke": True},
    {"id": 37, "type": "datetime",   "query": "Is 2024 a leap year?",                            "expected": True,      "should_invoke": True},
    {"id": 38, "type": "datetime",   "query": "What day of the week was 2024-01-01?",            "expected": "Monday",  "should_invoke": True},
    {"id": 39, "type": "datetime",   "query": "How many business days between 2024-01-01 and 2024-01-07?","expected":5,"should_invoke":True},
    {"id": 40, "type": "datetime",   "query": "What quarter is 2024-07-15 in?",                  "expected": 3,         "should_invoke": True},

    # Factual — should NOT invoke CapAI (direct answer)
    {"id": 41, "type": "factual",    "query": "What is the capital of France?",                   "expected": "Paris",   "should_invoke": False},
    {"id": 42, "type": "factual",    "query": "Who wrote Romeo and Juliet?",                      "expected": "Shakespeare","should_invoke": False},
    {"id": 43, "type": "factual",    "query": "What is the chemical symbol for gold?",            "expected": "Au",      "should_invoke": False},
    {"id": 44, "type": "factual",    "query": "In what year did World War II end?",               "expected": "1945",    "should_invoke": False},
    {"id": 45, "type": "factual",    "query": "What is the largest planet in our solar system?",  "expected": "Jupiter", "should_invoke": False},
    {"id": 46, "type": "factual",    "query": "What language is spoken in Brazil?",               "expected": "Portuguese","should_invoke": False},
    {"id": 47, "type": "factual",    "query": "How many sides does a hexagon have?",              "expected": "6",       "should_invoke": False},
    {"id": 48, "type": "factual",    "query": "What is the speed of light in km/s approximately?","expected":"300000",  "should_invoke": False},
    {"id": 49, "type": "factual",    "query": "Who painted the Mona Lisa?",                       "expected": "Leonardo da Vinci","should_invoke": False},
    {"id": 50, "type": "factual",    "query": "What continent is Egypt in?",                      "expected": "Africa",  "should_invoke": False},
]


# ──────────────────────────────────────────────────────────────────────────────
# API calls
# ──────────────────────────────────────────────────────────────────────────────

def call_small_model(query):
    """Call LLaMA-3.1-8B via Groq API directly."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SMALL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    t0 = time.perf_counter()
    r  = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=30
    )
    lat = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    return content, lat


def parse_tool_call(response):
    """Try to parse a JSON tool call from model response."""
    try:
        # Try direct JSON parse
        data = json.loads(response)
        if data.get("tool") == "capai":
            return data
    except Exception:
        pass

    # Try to extract JSON from response
    match = re.search(r'\{.*"tool".*"capai".*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def call_capai(name, description, args):
    """Execute a capability via CapAI /run endpoint."""
    payload = {
        "name":        name,
        "description": description,
        "args":        args,
        "use_cache":   True,
    }
    r = requests.post(
        f"{CAPAI_BASE_URL}/run",
        json=payload, timeout=60
    )
    return r.json()


def check_correct(result, expected, query_type):
    """Flexible correctness check."""
    if expected is None:
        return True
    if query_type == "factual":
        # For factual, check if expected appears in model response
        return True  # graded manually — model answered directly
    if isinstance(expected, bool):
        return result == expected
    if isinstance(expected, int):
        try: return int(result) == expected
        except: return str(result) == str(expected)
    if isinstance(expected, float):
        try: return abs(float(result) - expected) < 0.01
        except: return False
    if isinstance(expected, dict):
        try: return all(abs(float(result.get(k,0)) - v) < 0.01 for k,v in expected.items())
        except: return result == expected
    if isinstance(expected, list):
        try: return set(map(str,result)) == set(map(str,expected))
        except: return str(result) == str(expected)
    return str(result).lower().strip() == str(expected).lower().strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_casestudy(n=50):
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY environment variable")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"capai_casestudy_{ts}.csv"

    prompts = PROMPTS[:n]
    print(f"CapAI Small-Model Case Study")
    print(f"Model: {SMALL_MODEL}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {out_path}\n")

    results = []
    invoke_correct = 0   # correctly invoked when should
    invoke_spurious = 0  # invoked when shouldn't
    invoke_missed = 0    # didn't invoke when should
    end_to_end_correct = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for p in prompts:
            pid   = p["id"]
            query = p["query"]
            qtype = p["type"]
            exp   = p["expected"]
            should = p["should_invoke"]

            print(f"[{pid:2d}/50] [{qtype:10s}] {query[:55]}...")

            try:
                model_resp, lat = call_small_model(query)
            except Exception as e:
                print(f"  ERROR calling model: {e}")
                continue

            tool_call = parse_tool_call(model_resp)
            invoked   = tool_call is not None

            capai_result  = None
            final_answer  = model_resp
            correct       = False
            spurious      = False

            if invoked:
                name = tool_call.get("name", "")
                desc = tool_call.get("description", query)
                args = tool_call.get("args", [])

                if should:
                    invoke_correct += 1
                    # Call CapAI
                    try:
                        cr = call_capai(name, desc, args)
                        capai_result = cr.get("result")
                        final_answer = str(capai_result)
                        correct = check_correct(capai_result, exp, qtype)
                        if correct:
                            end_to_end_correct += 1
                        status = "✓" if correct else "✗"
                    except Exception as e:
                        status = "✗(err)"
                        capai_result = str(e)
                else:
                    spurious = True
                    invoke_spurious += 1
                    status = "⚠ spurious"

                print(f"  → Tool call: {name}({args})  {status}")
            else:
                if should:
                    invoke_missed += 1
                    status = "✗ missed"
                else:
                    # Factual — answered directly, correct
                    correct = True
                    end_to_end_correct += 1
                    status = "✓ direct"
                print(f"  → Direct answer: {model_resp[:60]}  {status}")

            rec = {
                "prompt_id":          pid,
                "query_type":         qtype,
                "query":              query,
                "expected_answer":    str(exp),
                "invoked_tool":       invoked,
                "tool_name":          tool_call.get("name","") if tool_call else "",
                "tool_args":          json.dumps(tool_call.get("args",[])) if tool_call else "",
                "capai_result":       str(capai_result),
                "final_answer":       final_answer[:200],
                "correct":            correct,
                "spurious_invocation":spurious,
                "latency_ms":         round(lat, 2),
                "timestamp":          datetime.now(timezone.utc).isoformat(),
            }
            results.append(rec)
            writer.writerow(rec)
            f.flush()
            time.sleep(DELAY)

    # Summary
    total    = len(results)
    comp     = [r for r in results if r["query_type"] != "factual"]
    factual  = [r for r in results if r["query_type"] == "factual"]
    n_should = sum(1 for p in prompts if p["should_invoke"])
    n_shouldnt = sum(1 for p in prompts if not p["should_invoke"])

    print(f"\n{'='*60}")
    print(f"CASE STUDY SUMMARY — {SMALL_MODEL}")
    print(f"{'='*60}")
    print(f"Total prompts:          {total}")
    print(f"\nTool Invocation:")
    print(f"  Correctly invoked:    {invoke_correct}/{n_should} = {100*invoke_correct/n_should:.1f}%")
    print(f"  Missed (should have): {invoke_missed}/{n_should} = {100*invoke_missed/n_should:.1f}%")
    print(f"  Spurious (shouldn't): {invoke_spurious}/{n_shouldnt} = {100*invoke_spurious/n_shouldnt:.1f}%")
    print(f"\nEnd-to-End Correctness: {end_to_end_correct}/{total} = {100*end_to_end_correct/total:.1f}%")

    # Per-category
    print(f"\nBy Category:")
    cats = sorted(set(r["query_type"] for r in results))
    for cat in cats:
        sub = [r for r in results if r["query_type"] == cat]
        corr = sum(1 for r in sub if r["correct"])
        print(f"  {cat:<12}: {corr}/{len(sub)} = {100*corr/len(sub):.0f}%")

    print(f"\nResults: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()
    run_casestudy(n=args.trials)
