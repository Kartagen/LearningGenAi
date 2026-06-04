"""LLM-as-a-judge evaluator using OpenRouter API."""
import json
import os
from typing import Any

import opik
from opik import opik_context
from opik.evaluation.metrics import base_metric, score_result
from openai import OpenAI


DEFAULT_JUDGE_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


YODA_EXAMPLE = (
    "The very Republic is threatened, if involved the Sith are. Hard to see, "
    "the dark side is. Discover who this assassin is, we must. With this Naboo "
    "queen you must stay, Qui-Gon. Protect her. May the Force be with you. "
    "A vergence, you say? But you do! Revealed your opinion is. Trained as a "
    "Jedi, you request for him? Good, good, young one."
)

SYSTEM_PROMPT_BASIC = """
You are an impartial judge that evaluates if text was written by {style}.

An example piece of text from {style} is:
{example}

Now, analyze some new text carefully and respond on if it follows the
same style of {style}. Be critical to identify any issues in the text.
Then convert your feedback into a number between 0 and 10: 10 if the text
is written exactly in the style of {style}, 5 if mixed faithfulness to the
style, or 0 if the text is not at all written in the style of {style}.

Directly answer with the score formatted in a dictionary.
The format of your response should only be the dictionary and nothing else:
{{"score": <score between 0 and 10>}}
"""


SYSTEM_PROMPT_DETAILED = """
You are an extremely strict linguistic judge that scores how closely a given text
follows the speech style of {style} from Star Wars.

Reference example of {style}'s speech:
{example}

Evaluate the candidate text against these specific style features:
  1. Object-Subject-Verb (OSV) inversion (e.g. "Powerful you have become").
  2. Fronted complements / topicalization ("Reckless is he").
  3. Short, sentence-fragment style with pauses and "yes", "mmm", "hmm".
  4. Wise / mystical tone (Force, Jedi, balance, fear, patience).
  5. Avoidance of modern colloquial English.

Penalize:
  - Standard SVO English (subject first, verb second).
  - Modern slang, jokes, lists, bullet points.
  - Direct factual answers without the inversion pattern.

Output ONLY a JSON dict like {{"score": <int 0..10>}} where:
  10 = perfect Yoda speech with multiple inversions,
   5 = some inversion / wise tone but mostly standard English,
   0 = ordinary modern English with no Yoda traits.
"""


class LLMClient:
    """Minimal OpenRouter chat client."""

    def __init__(self, model: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def ask(self, user: str, system: str | None = None, **kwargs):
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )


class LLMJudgeEvaluator(base_metric.BaseMetric):
    """Opik metric that delegates style scoring to an LLM judge."""

    def __init__(self, judge: LLMClient, system_prompt: str, max_score: int = 10):
        self.judge = judge
        self.system_prompt = system_prompt
        self.max_score = max_score
        self.prompt_template = "Evaluate this text: {text}"

    def score(self, text: str, n_tries: int = 5, **kwargs) -> score_result.ScoreResult:  # type: ignore[override]
        last_err: Exception | None = None
        for attempt in range(n_tries):
            try:
                prompt = self.prompt_template.format(text=text)
                with opik.start_as_current_span("judge_call"):
                    res = self.judge.ask(
                        system=self.system_prompt,
                        user=prompt,
                        max_tokens=50,
                    )
                    raw = res.choices[0].message.content.strip()
                    opik_context.update_current_span(
                        input={"system": self.system_prompt, "user": prompt},
                        output={"judge_raw_response": raw},
                        metadata={"phase": "judge"},
                        model=self.judge.model,
                        provider="openrouter",
                    )

                # robust JSON extraction
                start = raw.find("{")
                end = raw.rfind("}") + 1
                payload = json.loads(raw[start:end])
                score = float(payload["score"]) / self.max_score
                score = max(0.0, min(score, 1.0))
                return score_result.ScoreResult(name="StyleScore", value=score)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"Judge failed after {n_tries} attempts: {last_err}")


def make_yoda_judge(api_key: str, model_name: str = DEFAULT_JUDGE_MODEL, detailed: bool = False) -> LLMJudgeEvaluator:
    template = SYSTEM_PROMPT_DETAILED if detailed else SYSTEM_PROMPT_BASIC
    system_prompt = template.format(style="Yoda", example=YODA_EXAMPLE)
    client = LLMClient(model=model_name, api_key=api_key)
    return LLMJudgeEvaluator(client, system_prompt=system_prompt)
