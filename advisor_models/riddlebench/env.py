"""RiddleBench domain environment for SkyRL training.

Provides RiddleBenchEnv for advisor-guided riddle/puzzle solving.

Flow (advisor mode):
  advisor sees riddle → generates <diagnosis><advice> → advice sent to student → student answers → binary reward

Null-advice support (proposal Task 4):
  When null_advice=True in extras, the advisor action is overridden with an empty string
  and the student receives no guidance. These ~15% of rows provide a counterfactual
  baseline within the GRPO training batch without any trainer modifications.
"""

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from ..env_base import BaseAdvisorEnv
from ..utils.tag_utils import extract_advice
from .config import (
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_INSTRUCTION,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_INSTRUCTION,
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    STUDENT_INSTRUCTION_NULL,
    extract_answer,
    compute_score,
)


class RiddleBenchEnv(BaseAdvisorEnv):
    """Environment for advisor-guided riddle/puzzle solving on RiddleBench."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)

        assert "riddle_type" in extras, "riddle_type field is required"
        self.riddle_type = extras["riddle_type"]
        # null_advice=True rows override advisor action with "" (counterfactual baseline)
        self.null_advice = extras.get("null_advice", False)

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        formatted = BASELINE_INSTRUCTION.format(problem=self.original_question)
        return [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ], formatted

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Build advisor prompt showing the student's initial wrong attempt."""
        formatted = ADVISOR_INSTRUCTION.format(
            problem=self.original_question,
            riddle_type=self.riddle_type,
            initial_response=self.original_response,
        )
        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ]
        return messages, formatted

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        if advisor_feedback:
            advice = extract_advice(advisor_feedback)
            formatted = STUDENT_INSTRUCTION.format(
                problem=self.original_question,
                advisor_feedback=advice,
            )
        else:
            # null-advice path: student gets no guidance
            formatted = STUDENT_INSTRUCTION_NULL.format(problem=self.original_question)

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ], formatted

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        # R_outcome: local exact match — no API call
        extracted = extract_answer(self.final_response)
        r_outcome = compute_score(extracted, self.ground_truth)

        if self.advisor_mode == "advisor" and self.action:
            # R_diag: LLM judge — did the diagnosis correctly identify the student's error?
            r_diag = self._compute_diag_reward(self.action)
            # R_adh: token F1 between <advice> and revised response — no API call
            r_adh = self._compute_adh_reward(self.action, self.final_response)
        else:
            # baseline mode or null-advice: no advisor output to evaluate
            r_diag, r_adh = 0.0, 0.0

        reward = self._compute_total_reward(r_outcome, r_diag, r_adh)
        return reward, True, {
            "extracted_answer": extracted,
            "r_outcome": r_outcome,
            "r_diag": r_diag,
            "r_adh": r_adh,
        }

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        info = self.reward_info or {}
        r_adh = info.get("r_adh", 0.0)
        metadata["other_info"] = (
            f"type={self.riddle_type} | null={self.null_advice} | "
            f"ans={info.get('extracted_answer', '')} | "
            f"R_out={info.get('r_outcome', 0.0)} | "
            f"R_diag={info.get('r_diag', 0.0)} | "
            f"R_adh={r_adh:.3f}"
        )
        return metadata

    def step(self, action: str):
        """Override to apply null-advice substitution before normal step logic."""
        if self.null_advice:
            action = ""
        return super().step(action)
