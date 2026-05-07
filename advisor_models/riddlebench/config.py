"""Configuration for RiddleBench domain.

Contains prompts and scoring functions for advisor-guided riddle/puzzle solving.
Advisor outputs structured <diagnosis><advice> tags (proposal Task 2).
"""

import re

# ---------------------------------------------------------------------------
# Advisor prompts
# ---------------------------------------------------------------------------

STUDENT_INITIAL_SYSTEM_PROMPT = """You are a student attempting to solve a riddle or puzzle. \
Make two independent attempts using different approaches.

Format your response exactly as:

Attempt 1:
[your first approach and answer]

Attempt 2:
[a genuinely different approach and answer]

Keep each attempt concise and end it with your final answer."""

ADVISOR_SYSTEM_PROMPT = """You are an expert advisor reviewing a student's multiple failed attempts at a riddle or puzzle. Your role is to diagnose the root cause of the student's error across their attempts and provide corrective guidance.

You MUST structure every response with these two XML tags in order:
<diagnosis>Identify the root cause of the student's error in 1-2 sentences. Be specific about what went wrong in their reasoning across the attempts.</diagnosis><advice>Provide specific step-by-step corrective guidance in 3-5 sentences. Do NOT give away the answer — guide the student to find it themselves.</advice>"""

ADVISOR_INSTRUCTION = """A student made multiple attempts to solve the following riddle but could not get the correct answer.

Riddle: {problem}

Riddle type: {riddle_type}

Student's attempts:
{initial_response}

Diagnose the root cause of the student's errors and provide corrective guidance. Your response MUST use this format:
<diagnosis>Root cause of the student's error</diagnosis><advice>Step-by-step corrective guidance</advice>"""

# ---------------------------------------------------------------------------
# Student prompts (advisor mode)
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = """You are a student solving riddles and puzzles. You will receive a riddle and guidance from your advisor. Follow the guidance carefully to reason through the problem and arrive at the correct answer. Always end your response with the final answer in \\boxed{} format."""

STUDENT_INSTRUCTION = """Riddle: {problem}

Advisor Guidance:
{advisor_feedback}

Follow the guidance above to solve this riddle. Show your reasoning step by step, then provide your final answer in \\boxed{{}} format."""

STUDENT_INSTRUCTION_NULL = """Riddle: {problem}

Solve this riddle step by step and provide your final answer in \\boxed{{}} format."""

# ---------------------------------------------------------------------------
# Baseline prompts (baseline/direct RL mode)
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """You are an expert puzzle solver. Analyze the given riddle carefully and provide a well-reasoned solution. Always end with the final answer in \\boxed{} format."""

BASELINE_INSTRUCTION = """Riddle: {problem}

Solve this riddle step by step and provide your final answer in \\boxed{{}} format."""

# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------


def extract_answer(response: str) -> str:
    """Extract the final answer from a student response.

    Tries \\boxed{} first, then 'Answer:' pattern, then last non-empty line.
    """
    if not response:
        return ""

    # 1. \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        return m.group(1).strip()

    # 2. "Answer: ..." or "The answer is ..."
    m = re.search(r"(?:answer\s*(?:is\s*)?|final answer\s*(?:is\s*)?)[:\s]+([^\n.]+)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;")

    # 3. Last non-empty line as fallback
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation."""
    return re.sub(r"[.,;:!?\s]+$", "", s.lower().strip())


def compute_score(extracted_answer: str, ground_truth: str) -> float:
    """Return 1.0 if extracted answer matches ground truth, else 0.0."""
    if not extracted_answer or not ground_truth:
        return 0.0
    return 1.0 if _normalize(extracted_answer) == _normalize(ground_truth) else 0.0
