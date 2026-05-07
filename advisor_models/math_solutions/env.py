"""Math solutions domain environment for SkyRL training.

Provides MathSolutionsEnv class for math solution writing with advisor feedback.
Evaluates generated solutions for style alignment.
"""

import os
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
from openai import OpenAI

from ..env_base import BaseAdvisorEnv
from .config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    STYLE_JUDGE_SYSTEM_PROMPT,
    STYLE_JUDGE_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_INSTRUCTION,
)


class MathSolutionsEnv(BaseAdvisorEnv):
    """Environment for math solution writing with advisor feedback using 2-step flow."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)

        # additional required fields
        assert "student" in extras, "student field is required"
        assert "math_correct_answer" in extras, "math_correct_answer field is required"

        self.student = extras["student"]
        self.math_correct_answer = extras["math_correct_answer"]

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Build prompt for baseline model to solve the math problem."""
        formatted_prompt = BASELINE_INSTRUCTION.format(
            problem=self.original_question, student=self.student
        )
        return [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt},
        ], formatted_prompt

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        return super()._build_advisor_prompt(prompt)

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        """Build prompt for student model to solve the math problem."""
        formatted_prompt = STUDENT_INSTRUCTION.format(
            problem=self.original_question,
            advisor_feedback=advisor_feedback,
        )

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt},
        ], formatted_prompt

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        try:
            judge_prompt = STYLE_JUDGE_PROMPT.format(
                judge_criteria=self.ground_truth,
                solution=self.final_response,
            )

            judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
            api_base = os.environ.get("API_BASE", None)
            api_key = os.environ.get("OPENAI_API_KEY", None)
            client = OpenAI(base_url=api_base, api_key=api_key)
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": STYLE_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
            )

            judge_response = response.choices[0].message.content.strip()

            # Parse the three-way response: ACCEPT=1.0, PARTIAL=0.4, REJECT=0.0
            if "ACCEPT" in judge_response:
                return 1.0, True, {}
            elif "PARTIAL" in judge_response:
                return 0.4, True, {}
            else:  # REJECT or any other response
                return 0.0, True, {}

        except Exception as e:
            print(f"Error computing style reward: {e}")
            return 0.0, True, {}

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        metadata["other_info"] = (
            f"Student: {self.student}\nCorrect Answer: {self.math_correct_answer}"
        )
        return metadata
