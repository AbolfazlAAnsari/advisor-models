"""Base environment class for advisor-based RL training.

Provides BaseAdvisorEnv class that supports both advisor and baseline modes.
Subclasses implement domain-specific logic for different tasks.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig
from litellm import completion
import os

from .utils.tag_utils import extract_diagnosis, extract_advice

# ENVIRONMENT VARIABLES
ADVISOR_MODELS_MODE = os.environ.get("ADVISOR_MODELS_MODE", "advisor").lower()
"""Used to control the advisor models setting. Supported modes:
- `advisor`: Advisor model generates advice for student model to generate final response.
- `baseline`: Direct RL on the "advisor" model.
"""

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "gpt-4o-mini")
"""Used to set the student model. Overwritten if `model` is provided in dataset."""

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
"""Used to set the judge model for R_diag evaluation."""

API_BASE = os.environ.get("API_BASE", None)
"""Used to set the API base URL."""

DIAG_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator of advisor feedback for problem-solving tasks. "
    "Your task is to determine whether the advisor's diagnosis correctly "
    "identifies the root cause of the student's error."
)

DIAG_JUDGE_PROMPT = """Problem: {problem}

Student's incorrect answer: {initial_response}

Correct answer: {ground_truth}

Advisor's diagnosis: {diagnosis}

Does the advisor's diagnosis correctly identify the root cause of the student's error?
Briefly reason about whether the diagnosis accurately explains why the student failed, \
then end your response with exactly CORRECT or INCORRECT."""


class BaseAdvisorEnv(BaseTextEnv):
    """Base environment for Advisor Models RL training.

    Subclasses must implement:
    - `_build_baseline_prompt()`: For direct RL mode
    - `_build_student_prompt()`: For advisor mode student prompts
    - `_compute_step()`: For reward calculation

    Optionally override:
    - `_build_advisor_prompt()`: Default uses dataset prompt as-is
    - `_get_metadata()`: Default logs all standard fields
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """Initialize the environment.

        Args:
            env_config: The environment configuration.
            extras: Additional metadata.
        """
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras

        self.env_config = env_config
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]
        self.original_response = extras.get("original_response", "")
        self.initial_reward = extras.get("initial_reward", "")
        self.advisor_mode = ADVISOR_MODELS_MODE
        self.student_model = extras.get("model", STUDENT_MODEL)
        self.advisor_prompt_to_log = None
        self.action = None
        self.student_prompt_to_log = None
        self.final_response = None
        self.reward_info = None

    # ------------------------------------------------------------------
    # Multi-component gated reward  (proposal §5.4)
    # R_total = α·R_outcome + 1[R_outcome>0]·(β·R_diag + γ·R_adh)
    # ------------------------------------------------------------------

    def _compute_diag_reward(self, action: str) -> float:
        """R_diag: LLM judge evaluating whether <diagnosis> correctly identifies the student's error.

        Returns 1.0 if the judge deems the diagnosis accurate, 0.0 otherwise.
        Returns 0.0 immediately if no diagnosis tag is present (no API call made).
        """
        diagnosis = extract_diagnosis(action)
        if not diagnosis:
            return 0.0
        judge_prompt = DIAG_JUDGE_PROMPT.format(
            problem=self.original_question,
            initial_response=self.original_response,
            ground_truth=self.ground_truth,
            diagnosis=diagnosis,
        )
        try:
            response = completion(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": DIAG_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                base_url=API_BASE,
            )
            judge_response = response.choices[0].message.content.strip()
            return 1.0 if "CORRECT" in judge_response.upper() else 0.0
        except Exception as e:
            print(f"[BaseAdvisorEnv] R_diag judge call failed: {e}")
            return 0.0

    def _compute_adh_reward(self, action: str, student_response: str) -> float:
        """R_adh: token-level F1 between <advice> content and the student's revised response.

        F1 = 2·|overlap| / (|advice_tokens| + |response_tokens|)
        Measures lexical overlap as a proxy for whether the student followed the advice.
        """
        advice = extract_advice(action)
        if not advice or not student_response:
            return 0.0
        advice_tokens = set(advice.lower().split())
        response_tokens = set(student_response.lower().split())
        overlap = len(advice_tokens & response_tokens)
        if overlap == 0:
            return 0.0
        return 2 * overlap / (len(advice_tokens) + len(response_tokens))

    def _compute_total_reward(
        self, r_outcome: float, r_diag: float, r_adh: float
    ) -> float:
        """Gated multi-component reward (proposal §5.4).

        R_total = α·R_outcome + 1[R_outcome > 0]·(β·R_diag + γ·R_adh)

        Weights read from env_config: outcome_weight (α=1.0), diag_weight (β=0.1),
        adh_weight (γ=0.1). The gate ensures diagnosis/adherence credit is only
        awarded when the advice actually produces a correct outcome.
        """
        alpha = self.env_config.get("outcome_weight", 1.0)
        beta = self.env_config.get("diag_weight", 0.1)
        gamma = self.env_config.get("adh_weight", 0.1)
        gate = 1.0 if r_outcome > 0.0 else 0.0
        return alpha * r_outcome + gate * (beta * r_diag + gamma * r_adh)

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt used for direct RL of open-source model.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            str: The prompt to log.
        """
        raise NotImplementedError

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt sent to the advisor model.

        By default uses the prompt provided in the dataset, logging
        the last turn's content.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            str: The advisor prompt to log.
        """
        return prompt, prompt[-1]["content"]

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt sent to the student model. 2-step vs 3-step flow patterns
        should be handled here by changing prompt format.

        Args:
            advisor_feedback: The advisor feedback.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the student model.
            str: The student prompt to log.
        """
        raise NotImplementedError

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Computes reward.

        When this function is called, `self.final_response` is available.

        Returns:
            float: The scalar reward.
            bool: Whether the episode is done.
            Dict[str, Any]: Additional reward information saved to `self.reward_info`.
        """
        raise NotImplementedError

    def _get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for logging.

        Only fields `original_question`, `initial_response`, `advisor_prompt`,
        `advisor_response`, `student_prompt`, `student_response`, `ground_truth`,
        `initial_reward`, `final_reward`, and `other_info` are used.
        When this function is called, `self.advisor_prompt_to_log`, `self.action`,
        `self.student_prompt_to_log`, `self.final_response`, `self.final_reward`, and
        `self.reward_info` are available.
        Defaults to filling in all fields except `other_info`.

        Returns:
            Dict[str, Any]: Metadata dict for logging.
        """
        # handle student prompt/response
        if self.advisor_mode == "advisor":
            student_prompt = self.student_prompt_to_log
            student_response = self.final_response
        elif self.advisor_mode == "baseline":
            student_prompt = ""
            student_response = ""
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        return {
            "add_to_log": True,
            "original_question": self.original_question,
            "initial_response": self.original_response,
            "advisor_prompt": self.advisor_prompt_to_log,
            "advisor_response": self.action,
            "student_prompt": student_prompt,
            "student_response": student_response,
            "ground_truth": self.ground_truth,
            "initial_reward": self.initial_reward,
            "final_reward": self.final_reward,
            "other_info": "",
        }

    def init(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """SkyRL initialization for obtaining actual input prompt.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            Dict[str, Any]: Additional metadata (left empty).
        """

        if self.advisor_mode == "advisor":
            modified_prompt, self.advisor_prompt_to_log = self._build_advisor_prompt(
                prompt
            )
        elif self.advisor_mode == "baseline":
            modified_prompt, self.advisor_prompt_to_log = self._build_baseline_prompt(
                prompt
            )
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        return modified_prompt, {}

    def call_student(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> str:
        """Call the chat completion endpoint using student model.

        Args:
            messages: The messages to send to the student model.
            temperature: The temperature to use for the student model.
            timeout: Timeout in seconds for the API call (default: 120.0).

        Returns:
            str: The parsed response from the student model.
        """
        try:
            response = completion(
                model=self.student_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                base_url=API_BASE,
            )
            return response.choices[0].message.content
        except Exception as e:  # pragma: no cover
            print(f"[BaseAdvisorEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step of the environment.

        Args:
            action: The action taken by the policy model.

        Returns:
            BaseTextEnvStepOutput: The step output.
        """
        self.action = action
        if self.advisor_mode == "advisor":
            student_prompt, self.student_prompt_to_log = self._build_student_prompt(
                self.action
            )
            self.final_response = self.call_student(student_prompt, temperature=0.0)
        elif self.advisor_mode == "baseline":
            self.final_response = action
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        # Extract answer and compute reward
        self.final_reward, done, self.reward_info = self._compute_step()

        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.final_reward,
            done=done,
            metadata=self._get_metadata(),
        )
