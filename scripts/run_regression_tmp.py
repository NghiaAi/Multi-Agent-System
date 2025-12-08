import argparse
import json
import importlib
import os
import re
import time
from pathlib import Path
import sys

from pypdf import PdfReader

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_pdf_questions(pdf_path: Path, start: int, end: int):
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    text = re.sub(r"\r", "", text)
    lines = text.split("\n")

    questions = []
    current = None
    for line in lines:
        m = re.match(r"^\s*(\d+)\.\s*(.*)", line)
        if m:
            if current:
                questions.append(current)
            current = {"id": int(m.group(1)), "question": m.group(2).strip(), "answer": None}
            continue
        if line.strip().startswith("Answer:") and current:
            current["answer"] = line.split("Answer:", 1)[1].strip()

    if current:
        questions.append(current)

    questions = [q for q in questions if start <= q["id"] <= end]
    return questions


def fresh_agent():
    """
    Set GROQ_API_KEY and reload text_to_sql_agent to ensure the model picks the new key.
    """
    import agents.text_to_sql_agent as ttsa

    os.environ["GROQ_API_KEY"] = CURRENT_KEY[0]
    importlib.reload(ttsa)
    return ttsa.agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="First question id (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="Last question id (inclusive)")
    args = parser.parse_args()

    pdf_path = Path("Data_Platforms_Project_Regression_Test (1).pdf")
    questions = parse_pdf_questions(pdf_path, args.start, args.end)
    if not questions:
        raise SystemExit(f"No questions found in range {args.start}-{args.end}")
    print(f"Parsed {len(questions)} questions from PDF (ids {args.start}-{args.end})")

    # Prepare initial agent with first key
    agent = fresh_agent()

    results = []
    for q in questions:
        key_idx = KEY_TO_IDX.get(CURRENT_KEY[0], -1)
        prompt = q["question"] + "\nPlease answer shortly: Answer only."
        start_t = time.time()
        try:
            if hasattr(agent, "memory"):
                agent.memory.clear()
            resp = agent.run(prompt, stream=False, execute_tools=True)
            content = getattr(resp, "content", str(resp))
        except Exception as e:
            # On error (e.g., quota), rotate to next key and retry once
            try:
                CURRENT_KEY[0] = next(KEY_CYCLE)
                key_idx = KEY_TO_IDX.get(CURRENT_KEY[0], -1)
                agent = fresh_agent()
                if hasattr(agent, "memory"):
                    agent.memory.clear()
                resp = agent.run(prompt, stream=False, execute_tools=True)
                content = getattr(resp, "content", str(resp))
            except Exception as ee:
                content = f"ERROR: {e} | retry: {ee}"
        elapsed = time.time() - start_t
        answer_line = ""
        if isinstance(content, str):
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            # Ưu tiên dòng bắt đầu bằng "Answer:"
            ans_lines = [ln for ln in lines if ln.lower().startswith("answer:")]
            if ans_lines:
                answer_line = ans_lines[0]
            elif lines:
                # Nếu không có "Answer:" lấy dòng cuối cùng không rỗng
                answer_line = lines[-1]
        else:
            answer_line = str(content)

        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer_expected": q.get("answer"),
                "answer_model": answer_line,
                "secs": round(elapsed, 2),
                "key_index": key_idx,
                "key_last6": CURRENT_KEY[0][-6:] if CURRENT_KEY and CURRENT_KEY[0] else "",
            }
        )
        preview = answer_line[:120]
        print(f"#{q['id']:3d} took {elapsed:.1f}s -> {preview}")

    out_path = Path(f"scripts/regression_results_{args.start}_{args.end}.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    # Load keys and prepare cycle before main()
    keys_path = Path("key.json")
    keys = json.loads(keys_path.read_text()).get("GROQ_KEYS", [])
    if not keys:
        raise SystemExit("No GROQ_KEYS found in key.json")
    KEY_LIST = keys
    KEY_TO_IDX = {k: i for i, k in enumerate(KEY_LIST)}
    KEY_CYCLE = iter(KEY_LIST[1:] + KEY_LIST[:1])  # rotation iterator after first key
    CURRENT_KEY = [KEY_LIST[0]]
    main()

