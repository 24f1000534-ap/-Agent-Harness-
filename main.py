from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI()


# ---------- 1. Define what the incoming JSON looks like ----------

class Step(BaseModel):
    step_number: int
    tool: str
    args: Dict[str, Any]
    tokens_used: int


class RunRequest(BaseModel):
    budget_tokens: int
    steps: List[Step] = []


# ---------- 2. Helper: make two "same looking" args objects actually equal ----------

def canonicalize(value: Any) -> Any:
    """
    Turns args into a normalized form so that two calls that are
    'basically the same' compare equal:
      - drop the 'trace_id' key anywhere in the object (it changes every call on purpose)
      - collapse/trim whitespace inside string values
      - key ORDER doesn't matter because Python dict equality already
        ignores order -- we don't need to sort for that reason, we only
        need to strip trace_id and normalize whitespace.
    """
    if isinstance(value, dict):
        return {
            k: canonicalize(v)
            for k, v in value.items()
            if k != "trace_id"
        }
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.split())  # collapses all whitespace runs, trims ends
    return value


def call_signature(step: Step):
    """A hashable/comparable fingerprint of one step: (tool, canonical args)."""
    return (step.tool, canonicalize(step.args))


# ---------- 3. The actual policy engine ----------

def decide(req: RunRequest) -> Dict[str, str]:
    steps = req.steps
    n = len(steps)

    # --- Rule A: empty history -> nothing to judge yet ---
    if n == 0:
        return {"decision": "continue", "reason": "Fresh run, no steps taken yet."}

    # --- Rule B: budget check (sum tokens_used across ALL steps) ---
    total_tokens = sum(s.tokens_used for s in steps)
    if total_tokens >= req.budget_tokens:
        return {
            "decision": "halt",
            "reason": f"Cumulative tokens_used ({total_tokens}) has reached "
                      f"the budget ({req.budget_tokens}).",
        }

    # --- Rule C: 3-or-more identical calls in a row (looking backward from the end) ---
    last_sig = call_signature(steps[-1])
    run_length = 1
    i = n - 2
    while i >= 0 and call_signature(steps[i]) == last_sig:
        run_length += 1
        i -= 1
    if run_length >= 3:
        return {
            "decision": "halt",
            "reason": f"Tool '{steps[-1].tool}' called with identical arguments "
                      f"{run_length} times in a row -- looks like a stuck loop.",
        }

    # --- Rule D: 2-step alternating cycle A,B,A,B,A,B across the trailing 6+ steps ---
    if n >= 6:
        tail = steps[-6:]
        sigs = [call_signature(s) for s in tail]
        a, b = sigs[0], sigs[1]
        is_cycle = (
            a != b
            and sigs[0] == sigs[2] == sigs[4]
            and sigs[1] == sigs[3] == sigs[5]
        )
        if is_cycle:
            return {
                "decision": "halt",
                "reason": f"Trailing steps show a repeating A/B cycle between "
                          f"'{tail[0].tool}' and '{tail[1].tool}' with no new "
                          f"progress -- looks like a stuck loop.",
            }

    # --- Nothing tripped: safe to continue ---
    return {
        "decision": "continue",
        "reason": f"Under budget ({total_tokens}/{req.budget_tokens} tokens) "
                  f"and no repeated-call or cycle pattern detected.",
    }


# ---------- 4. The endpoint itself ----------

@app.post("/run-budget-loop-guard")
def run_budget_loop_guard(req: RunRequest):
    return decide(req)


@app.get("/")
def health():
    return {"status": "ok"}
