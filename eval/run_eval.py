"""Eval-сет ask_notes (модуль C): прогон вопросов через run_api и метрики.

Метрики:
- retrieval_hit: ожидаемый файл есть в блоке «Источники» (слой поиска);
- cited_hit: ожидаемый файл среди файлов процитированных чанков (поиск + выбор цитат);
- verdict (LLM-as-a-judge): supported / partial / hallucinated / refusal;
- refusal_ok для негативных вопросов;
- token spend (prompt/completion из usage пайплайна + usage судьи), время ответа.

Запуск (из digital-engineer-assistant):
    .venv/Scripts/python.exe eval/run_eval.py            # полный прогон
    .venv/Scripts/python.exe eval/run_eval.py --limit 2  # быстрый тест
    .venv/Scripts/python.exe eval/run_eval.py --skip-judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "ask_notes"))

import pipeline_wrapper as pw  # noqa: E402

DATASET_PATH = Path(__file__).with_name("dataset.json")
JUDGE_MODEL = "google/gemma-4-e4b"
JUDGE_TIMEOUT_S = 120.0

REFUSAL_MARKERS = (
    "не нашлось",
    "не найдено",
    "не обнаружено",
    "нет информации",
    "нет в документах",
    "в предоставленных документах",
    "сервис временно недоступен",
    "не удалось найти",
    "нет данных по",
)

CITATION_RE = re.compile(r"\[(chk_[A-Za-z0-9]+)\]")
SOURCES_BLOCK_RE = re.compile(r"Источники:\n(.*)$", re.S)
SOURCE_LINE_RE = re.compile(r"^- (.+?) \[(chk_[A-Za-z0-9]+)\]$", re.M)

JUDGE_SYSTEM = (
    "Ты строгий проверщик ответов RAG-ассистента. Сравни ответ ассистента с эталоном. "
    "Вердикты: supported (ответ полностью совпадает с эталоном по факту), "
    "partial (факт тот же, но есть неточности или лишнее), "
    "hallucinated (ассистент выдумал факты, противоречит эталону или ответил на вопрос, "
    "которого нет в базе, когда эталон говорит об отказе), "
    "refusal (ассистент явно отказался отвечать, ссылаясь на отсутствие данных в документах). "
    "Отвечай СТРОГО одним JSON-объектом: "
    '{"verdict": "...", "reason": "коротко на русском"}.'
)


def split_answer_and_sources(output: str) -> tuple[str, list[tuple[str, str]]]:
    """Разбирает вывод run_api: (текст ответа, [(source_path, chunk_id)])."""
    match = SOURCES_BLOCK_RE.search(output)
    if not match:
        return output.strip(), []
    answer = output[: match.start()].strip()
    sources = SOURCE_LINE_RE.findall(match.group(1))
    return answer, sources


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def judge(client: Any, question: str, expected: str, answer: str) -> dict[str, Any]:
    user = (
        f"Вопрос: {question}\n\n"
        f"Эталонный ответ: {expected}\n\n"
        f"Ответ ассистента: {answer}"
    )
    resp = None
    text = ""
    for _attempt in (1, 2):
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
            timeout=JUDGE_TIMEOUT_S,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            break
    usage = resp.usage if resp is not None else None
    finish = resp.choices[0].finish_reason if resp is not None else "no-response"
    if not text:
        return {
            "verdict": "judge_error",
            "reason": f"пустой ответ судьи (finish_reason={finish})",
            "raw": "",
            "tokens": usage,
        }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {
            "verdict": "judge_error",
            "reason": "нет JSON в ответе судьи",
            "raw": text[:500],
            "tokens": usage,
        }
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return {
            "verdict": "judge_error",
            "reason": f"сломанный JSON судьи: {exc}",
            "raw": text[:500],
            "tokens": usage,
        }
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("supported", "partial", "hallucinated", "refusal"):
        verdict = "judge_error"
    return {"verdict": verdict, "reason": str(data.get("reason", "")), "tokens": usage}


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон eval-сета ask_notes")
    parser.add_argument("--limit", type=int, default=0, help="Сколько вопросов прогнать")
    parser.add_argument("--skip-judge", action="store_true", help="Без LLM-as-a-judge")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    wrapper = pw.PipelineWrapper()
    wrapper.setup()

    judge_client = None
    if not args.skip_judge:
        from openai import OpenAI

        judge_client = OpenAI(base_url=pw.LMSTUDIO_URL, api_key="lm-studio", timeout=JUDGE_TIMEOUT_S)

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected_answer"]
        expected_source = item.get("expected_source")
        is_positive = item["type"] == "positive"

        usage_before = dict(wrapper.usage)
        started = time.perf_counter()
        try:
            output = wrapper.run_api(question)
        except Exception as exc:  # noqa: BLE001 — прогон не должен валидаться на одном вопросе
            output = f"ОШИБКА ПРОГОНА: {type(exc).__name__}: {exc}"
        elapsed_s = round(time.perf_counter() - started, 1)
        usage_after = dict(wrapper.usage)

        answer, sources = split_answer_and_sources(output)
        source_paths = [path for path, _ in sources]
        cited_paths = [
            path for path, chunk_id in sources if chunk_id in CITATION_RE.findall(answer)
        ]
        refusal = is_refusal(answer)

        record: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "question": question,
            "expected_answer": expected,
            "expected_source": expected_source,
            "answer": answer,
            "cited_sources": cited_paths,
            "retrieved_sources": source_paths,
            "refusal": refusal,
            "elapsed_s": elapsed_s,
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }

        if is_positive:
            record["retrieval_hit"] = expected_source in source_paths if expected_source else False
            record["cited_hit"] = expected_source in cited_paths if expected_source else False
        else:
            record["refusal_ok"] = refusal

        if judge_client is not None:
            try:
                j = judge(judge_client, question, expected, answer)
            except Exception as exc:  # noqa: BLE001
                j = {"verdict": "judge_error", "reason": f"{type(exc).__name__}: {exc}", "tokens": None}
            usage = j.get("tokens")
            record["judge"] = {
                "verdict": j["verdict"],
                "reason": j["reason"],
                "raw": j.get("raw", ""),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }

        results.append(record)
        verdict = record.get("judge", {}).get("verdict", "-")
        print(
            f"[{idx}/{len(dataset)}] {item['id']} {elapsed_s}s "
            f"refusal={refusal} judge={verdict} q={question[:60]}",
            flush=True,
        )

    positives = [r for r in results if r["type"] == "positive"]
    negatives = [r for r in results if r["type"] == "negative"]
    judged = [r for r in results if "judge" in r]

    def rate(num: int, den: int) -> float:
        return round(num / den, 3) if den else 0.0

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": pw.CHAT_MODEL,
        "judge_model": None if args.skip_judge else JUDGE_MODEL,
        "questions": len(results),
        "positives": len(positives),
        "negatives": len(negatives),
        "retrieval_hit_rate": rate(sum(1 for r in positives if r.get("retrieval_hit")), len(positives)),
        "cited_hit_rate": rate(sum(1 for r in positives if r.get("cited_hit")), len(positives)),
        "refusal_ok_rate": rate(sum(1 for r in negatives if r.get("refusal_ok")), len(negatives)),
        "hallucination_rate": rate(
            sum(1 for r in judged if r["judge"]["verdict"] == "hallucinated"), len(judged)
        ),
        "supported_rate": rate(
            sum(1 for r in judged if r["judge"]["verdict"] == "supported"), len(judged)
        ),
        "judge_errors": sum(1 for r in judged if r["judge"]["verdict"] == "judge_error"),
        "pipeline_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "pipeline_completion_tokens": sum(r["completion_tokens"] for r in results),
        "judge_prompt_tokens": sum(r["judge"]["prompt_tokens"] for r in judged),
        "judge_completion_tokens": sum(r["judge"]["completion_tokens"] for r in judged),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / len(results), 1) if results else 0,
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in results), 1),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent
    (out_dir / f"results_{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results_latest.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
