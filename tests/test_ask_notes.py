"""Тесты MCP-инструмента ask_notes (pipeline_wrapper.PipelineWrapper.run_api).

Покрывают заявленный MCP-инструмент:
  ask_notes(question: str) -> str

Сценарии:
  - отказ на пустой/пробельный вопрос;
  - успешный ответ: текст + блок «Источники»;
  - пустой результат поиска: вежливый отказ;
  - падение всех попыток генерации: фолбэк;
  - валидация ответа (JSON + схема + цитаты): фолбэк после ретрая;
  - разбор JSON и проверка цитат (_parse_answer, _check_citations).

Запуск (из digital-engineer-assistant):
    .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "ask_notes"))

import pipeline_wrapper as pw  # noqa: E402


def _doc(chunk_id: str, source: str, content: str = "text") -> dict:
    """Собрать haystack Document без импорта haystack (минимум нужных полей)."""
    from haystack.dataclasses import Document

    return Document(content=content, meta={"chunk_id": chunk_id, "source": source, "heading": "", "score": 1.0})


class AskNotesQuestionValidation(unittest.TestCase):
    """Пустой/некорректный ввод: run_api обязан падать с ValueError, без обращений наружу."""

    def test_empty_question_raises(self) -> None:
        wrapper = pw.PipelineWrapper()
        with self.assertRaises(ValueError):
            wrapper.run_api("")

    def test_whitespace_only_question_raises(self) -> None:
        wrapper = pw.PipelineWrapper()
        with self.assertRaises(ValueError):
            wrapper.run_api("   \n\t  ")


class ParseAnswer(unittest.TestCase):
    """_parse_answer: вытащить JSON из ответа модели и валидировать схему."""

    def test_clean_json_parses(self) -> None:
        text = '{"answer": "42", "citations": ["chk_a"]}'
        parsed = pw.PipelineWrapper._parse_answer(text)
        self.assertEqual(parsed.answer, "42")
        self.assertEqual(parsed.citations, ["chk_a"])

    def test_json_embedded_in_text(self) -> None:
        text = 'Вот ответ: {"answer": "ok", "citations": ["chk_x"]} конец.'
        parsed = pw.PipelineWrapper._parse_answer(text)
        self.assertEqual(parsed.answer, "ok")

    def test_no_json_raises_validation_failure(self) -> None:
        with self.assertRaises(pw.ValidationFailure):
            pw.PipelineWrapper._parse_answer("модель не вернула JSON")

    def test_broken_json_raises_validation_failure(self) -> None:
        with self.assertRaises(pw.ValidationFailure):
            pw.PipelineWrapper._parse_answer("{'answer': ")

    def test_wrong_schema_raises_validation_failure(self) -> None:
        with self.assertRaises(pw.ValidationFailure):
            pw.PipelineWrapper._parse_answer('{"answer": 123, "citations": []}')


class CheckCitations(unittest.TestCase):
    """_check_citations: все цитаты должны быть среди выданных chunk_id."""

    def test_all_citations_match(self) -> None:
        docs = [_doc("chk_a", "gost.md"), _doc("chk_b", "reg.md")]
        parsed = pw.StructuredAnswer(answer="ok", citations=["chk_a", "chk_b"])
        pw.PipelineWrapper._check_citations(parsed, docs)

    def test_unknown_citation_raises(self) -> None:
        docs = [_doc("chk_a", "gost.md")]
        parsed = pw.StructuredAnswer(answer="ok", citations=["chk_zzz"])
        with self.assertRaises(pw.ValidationFailure):
            pw.PipelineWrapper._check_citations(parsed, docs)

    def test_empty_citations_raises(self) -> None:
        docs = [_doc("chk_a", "gost.md")]
        parsed = pw.StructuredAnswer(answer="ok", citations=[])
        with self.assertRaises(pw.ValidationFailure):
            pw.PipelineWrapper._check_citations(parsed, docs)

    def test_bracketed_citation_normalized(self) -> None:
        """Модель любит оборачивать chunk_id в [..] — должно сниматься."""
        docs = [_doc("chk_a", "gost.md")]
        parsed = pw.StructuredAnswer(answer="ok", citations=["[chk_a]"])
        pw.PipelineWrapper._check_citations(parsed, docs)


class WithSources(unittest.TestCase):
    """_with_sources: форматирует блок «Источники» в конце ответа."""

    def test_appends_sources_block(self) -> None:
        docs = [_doc("chk_a", "gost.md"), _doc("chk_b", "reg.md")]
        out = pw.PipelineWrapper._with_sources("ответ", docs)
        self.assertIn("ответ", out)
        self.assertIn("Источники:", out)
        self.assertIn("- gost.md [chk_a]", out)
        self.assertIn("- reg.md [chk_b]", out)


class RunApiBehavior(unittest.TestCase):
    """run_api: основные ветки с моком ретривера и генератора."""

    def _wrapper_with_mocks(self, documents, generated_text=None, generated_side_effect=None):
        """Собрать wrapper, подменив retriever.run и _call_with_deadline (LM Studio)."""
        wrapper = pw.PipelineWrapper()
        wrapper.setup()

        wrapper.retriever.run = lambda query: {"documents": documents}  # type: ignore[assignment]

        if generated_side_effect is not None:

            def fake_deadline(step, timeout_s, fn, *args, **kwargs):
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and "replies" in result:
                    text = next(generated_side_effect)
                    from haystack.dataclasses import ChatMessage

                    result["replies"] = [ChatMessage.from_assistant(text)]
                return result

            wrapper._call_with_deadline = fake_deadline  # type: ignore[assignment]
        return wrapper

    def test_happy_path_returns_answer_with_sources(self) -> None:
        docs = [_doc("chk_a", "gost.md", "ГОСТ 22.13330")]
        wrapper = self._wrapper_with_mocks(
            documents=docs,
            generated_side_effect=iter([json.dumps({"answer": "42", "citations": ["chk_a"]})]),
        )
        out = wrapper.run_api("что-то")
        self.assertIn("42", out)
        self.assertIn("Источники:", out)
        self.assertIn("gost.md [chk_a]", out)

    def test_empty_search_returns_polite_refusal(self) -> None:
        wrapper = self._wrapper_with_mocks(documents=[])
        out = wrapper.run_api("что-то")
        self.assertIn("не нашлось", out.lower())

    def test_validation_failure_falls_back_after_retry(self) -> None:
        """Модель дважды вернула мусор — должен сработать фолбэк, без crash."""
        docs = [_doc("chk_a", "gost.md")]
        wrapper = self._wrapper_with_mocks(
            documents=docs,
            generated_side_effect=iter(["это не JSON вообще", "тоже не JSON"]),
        )
        out = wrapper.run_api("вопрос")
        self.assertIn("временно недоступен", out.lower())

    def test_persistent_failure_falls_back(self) -> None:
        """_search кидает исключение на всех попытках — должен быть фолбэк."""
        wrapper = pw.PipelineWrapper()
        wrapper.setup()

        def boom(query):
            raise RuntimeError("mdrack упал")

        wrapper.retriever.run = boom  # type: ignore[assignment]
        out = wrapper.run_api("вопрос")
        self.assertIn("временно недоступен", out.lower())
        self.assertIn("mdrack", out.lower())


class ReasonTruncation(unittest.TestCase):
    """_reason: обрезка и формирование текста причины для фолбэка."""

    def test_short_message_kept(self) -> None:
        self.assertEqual(pw.PipelineWrapper._reason(RuntimeError("x")), "x")

    def test_long_message_truncated(self) -> None:
        long = "a" * 500
        out = pw.PipelineWrapper._reason(RuntimeError(long))
        self.assertLessEqual(len(out), 200)


if __name__ == "__main__":
    unittest.main()
