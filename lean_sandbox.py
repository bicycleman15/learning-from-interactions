#!/usr/bin/env python3
"""
lean_sandbox.py

A sandbox environment where the LLM decides what to prove and writes Lean 4 code.
The model has full control over theorem statements and proofs.

INSTALL:
  pip install openai anthropic

ENV:
  export OPENAI_API_KEY=...
  export ANTHROPIC_API_KEY=...

Edit the file problem.txt for the problem you want solved.
gpt-5.2 complains about some prompt policy, so we are using 5.1 for now.

USAGE:
  python lean_sandbox.py --provider openai --model gpt-5.2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple


# -----------------------------
# Logging schema (JSONL events)
# -----------------------------
EventType = Literal[
    "meta",
    "think",           # Agent reasoning
    "lean_action",     # Lean code executed
    "lean_result",     # Raw Lean output  
    "observation",     # Agent's interpretation of result
    "answer",          # Final answer without full verification
    "llm_io",
    "finished",        # Lean verification succeeded
    "crash",
]

@dataclass
class Event:
    problem_id: str
    event_type: EventType
    t: float
    payload: Dict[str, Any]


def append_jsonl(path: str, ev: Event) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


# -----------------------------
# Lean 4 Runner (with Mathlib project)
# -----------------------------
import random

def corrupt_numbers(text: str, probability: float = 0.5) -> str:
    """Randomly corrupt integers in the output by different amounts."""
    def maybe_corrupt(match):
        num = int(match.group(0))
        if random.random() < probability:
            # Random offset: -3 to +3, excluding 0
            offset = random.choice([-3, -2, -1, 1, 2, 3])
            return str(num + offset)
        return str(num)
    # Match integers (including negative), but avoid touching things like "x^3"
    return re.sub(r'-?\b\d+\b', maybe_corrupt, text)


class LeanRunner:
    """Run Lean 4 code using a pre-configured Mathlib project."""
    
    def __init__(self, timeout: int = 120, project_dir: str = None, corrupt_output: bool = False):
        self.timeout = timeout
        self.corrupt_output = corrupt_output
        # Use the lean_project directory with Mathlib
        if project_dir is None:
            self.project_dir = Path(__file__).parent / "lean_project"
        else:
            self.project_dir = Path(project_dir)
    
    def run_code(self, code: str) -> Tuple[bool, str]:
        """
        Run Lean 4 code using the Mathlib project and return (success, output).
        """
        # Write code to a scratch file in the project
        scratch_file = self.project_dir / "Scratch.lean"
        scratch_file.write_text(code)
        
        try:
            # Use lake env lean to run with proper dependencies
            result = subprocess.run(
                ["lake", "env", "lean", str(scratch_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.project_dir
            )
            
            if result.returncode == 0:
                output = result.stdout or "✓ Code compiled successfully!"
                if self.corrupt_output:
                    output = corrupt_numbers(output)
                return True, output
            else:
                return False, result.stderr or result.stdout or "Unknown error"
                
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {self.timeout}s"
        except FileNotFoundError:
            return False, "Lean 4 not found. Please install elan/lean4."
        except Exception as e:
            return False, f"Error: {e}"


# -----------------------------
# LLM adapters
# -----------------------------
class LLM:
    def decide(self, *, problem: str, transcript: str, lean_state: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def comment(self, *, problem: str, transcript: str, code: str, result: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError


class OpenAIResponsesLLM(LLM):
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.messages: List[Dict[str, str]] = []  # Accumulated conversation

    def _call_with_messages(self, *, max_output_tokens: int = 4000, temperature: float = 0.9) -> Tuple[str, Dict[str, Any]]:
        resp = self.client.responses.create(
            model=self.model,
            input=self.messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        raw = {"id": getattr(resp, "id", None)}
        # Append assistant response to conversation
        self.messages.append({"role": "assistant", "content": text})
        return text, raw

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_system_message(self, content: str) -> None:
        """Add a system message to start the conversation."""
        self.messages.append({"role": "system", "content": content})

    def decide(self, *, problem: str, transcript: str, lean_state: str, lean_used: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        # On first call, set up the system prompt
        if not self.messages:
            answer_note = "" if lean_used else " (requires at least one LEAN action first)"
            mode_block = (
                "Choose your next action:\n"
                "- THINK: <reason about the problem, plan, interpret observations>\n"
                "- LEAN:\n```lean\n<Lean 4 code to execute>\n```\n"
                f"- ANSWER: \\boxed{{<your final answer>}}{answer_note}\n"
            )
            system_prompt = f"""
I am trying to collect interaction logs with Lean for finetuning a smaller agent model.
Your job is to generate these interactions logs. 
In order to have useful interaction logs, the agent your are simulating should make slight errors in its "reasoning" or "interaction" with Lean, and then rectify them based on the observations (output logs) from Lean.

The final goal of the agent is to find the answer (which goes in \\boxed{{}}) using Lean 4 as kind of a computational or interaction tool.

In your simulation, the agent should have mistakes in between sometimes, so that there is diversity in interaction logs.
The agent should THINK about the problem first in the interaction logs.

IMPORTANT: The Agent you are simulating should NOT compute the final answer directly first! It should try to interact with lean to find the answer. It should use Lean for ALL arithmetic and computation.

HOW THE AGENT SHOULD USE LEAN FOR COMPUTATION, few examples:
- `#eval (125 : Rat) / 9` - rational arithmetic
- `#eval (2.5 : Float) * 3.0` - float arithmetic
- `#eval Float.sqrt 2` - square roots
- `#check <expr>` - check types
- `import Mathlib` - for ℝ, ℂ, trig, etc.

HOW THE AGENT YOU ARE SIMULATING SHOULD STRUCTURE ITS ACTIONS:
1. The agent should start with THINK to analyze the problem - identify what needs to be computed
2. Write LEAN code to perform calculations step by step
3. After each LEAN output, observe the result and decide next step
4. If there's an error, THINK about what went wrong, then try again
5. Once you have computed the final answer using Lean, output ANSWER with the result

HOW LEAN OUTPUT WILL BE PROVIDED:
- After the agent outputs a LEAN action, I will execute the code and provide the output
- The output will appear as "LEAN OUTPUT:" followed by the result
- If the code compiles successfully, you'll see the computed value or "✓ Code compiled successfully!"
- If there's an error, you'll see the error message which explains what went wrong
- The agent you are simulating should use this as feedback to guide its next action (the agent should THINK about the result, then continue)

EXAMPLE INTERACTION PATTERN:
THINK: I need to convert (0,3) to polar. r = sqrt(x² + y²), θ = atan2(y,x). Let me compute r first.

LEAN:
```lean
#eval Float.sqrt (0*0 + 3*3)
```

[After seeing output: 3.0]

THINK: r = 3. Now I need θ. Since x=0 and y>0, θ = π/2.

LEAN:
```lean
#eval Float.atan2 3 0
```

[After seeing output: 1.5707...]

THINK: θ ≈ 1.5708 which is π/2. So polar coordinates are (3, π/2).

ANSWER: \\boxed{{(3, \\frac{{\\pi}}{{2}})}}

Note that the agent you are simulating doesn't always have to use Lean.
The agent should use Lean whenever it feels like it needs it. 
Also, try to have both Lean and Math mistakes in your agent simulation, so that training data contains both of these rectifications.
But don't have too much mistakes, there should be a balance :)

Also, don't use Lean always, there should be some cases where agent solves the problem directly too without interaction with lean.
Don't let the agent do any sanity checks on top of Lean's outputs, they are always correct :)

Have some trajectories from which the agent corrects math mistakes from the Lean output log!

PROBLEM:
{problem}

Now begin the simulation. Output the agent's first action.
"""
            self.add_system_message(system_prompt)
        
        # Get the next response
        return self._call_with_messages()

    def add_observation(self, code: str, result: str, success: bool) -> None:
        """Add Lean execution result as a user message."""
        status = "✓ Success" if success else f"✗ Error"
        observation = f"LEAN OUTPUT:\n{result}"
        self.add_user_message(observation)


# -----------------------------
# Parsing
# -----------------------------
LEAN_BLOCK_RE = re.compile(r"```lean\n(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"^\s*THINK:\s*(.+)", re.DOTALL)
ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+)", re.DOTALL)
OBSERVATION_RE = re.compile(r"^\s*OBSERVATION:\s*(.+)", re.DOTALL)


def parse_decision(text: str, *, require_lean: bool) -> Tuple[str, str]:
    """
    Returns (kind, content) where kind in {"LEAN", "THINK", "ANSWER"}
    Priority: ANSWER > LEAN > THINK (if model gives answer, we're done)
    """
    text = text.strip()
    
    # Check for ANSWER first (highest priority - if model gives answer, we're done)
    # Search anywhere in text, not just at start
    answer_match = re.search(r"ANSWER:\s*(.+?)(?=\n\n|\Z)", text, re.DOTALL)
    if answer_match:
        return "ANSWER", answer_match.group(1).strip()
    
    # Check for Lean code block
    m = LEAN_BLOCK_RE.search(text)
    if m:
        return "LEAN", m.group(1).strip()
    
    # Check for LEAN: prefix without code block
    if text.upper().startswith("LEAN:"):
        code = text[5:].strip()
        # Remove any markdown markers
        code = code.replace("```lean", "").replace("```", "").strip()
        return "LEAN", code
    
    if require_lean:
        # In unblock mode, treat everything as attempted Lean code
        code = text.replace("```lean", "").replace("```", "").strip()
        if code:
            return "LEAN", code
        raise ValueError("Expected LEAN code, got empty output.")
    
    # Check for THINK prefix
    m = THINK_RE.match(text)
    if m:
        return "THINK", m.group(1).strip()
    
    # Default to thinking (natural text without prefix)
    return "THINK", text


def parse_observation(text: str) -> str:
    text = text.strip()
    m = OBSERVATION_RE.match(text)
    if m:
        return "OBSERVATION: " + m.group(1).strip()
    return "OBSERVATION: " + text if text else "OBSERVATION: (no observation)"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", required=False)
    ap.add_argument("--problem_file", default="problem.txt", help="Path to file containing the problem statement")
    ap.add_argument("--problem_id", default="problem_0001")
    ap.add_argument("--out", default="traj.jsonl")
    ap.add_argument("--pretty", default=None)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--log_llm_io", action="store_true")
    ap.add_argument("--corrupt", action="store_true", help="Add +1 to all numbers in Lean output")

    args = ap.parse_args()
    
    # Read problem from file
    problem_path = Path(args.problem_file)
    if not problem_path.exists():
        raise SystemExit(f"Problem file not found: {args.problem_file}")
    problem = problem_path.read_text().strip()

    model = args.model
    if args.provider == "openai" and not model:
        model = "gpt-5.2"
    if args.provider == "anthropic" and not model:
        raise SystemExit("For --provider anthropic you must pass --model")

    llm: LLM
    if args.provider == "openai":
        llm = OpenAIResponsesLLM(model=model)
    else:
        llm = AnthropicMessagesLLM(model=model)

    lean_runner = LeanRunner(timeout=args.timeout, corrupt_output=args.corrupt)

    # Meta header
    append_jsonl(
        args.out,
        Event(
            problem_id=args.problem_id,
            event_type="meta",
            t=time.time(),
            payload={
                "provider": args.provider,
                "model": model,
                "problem": problem,
                "max_steps": args.max_steps,
                "timeout": args.timeout,
                "mode": "sandbox",
            },
        ),
    )

    transcript_blocks: List[str] = []
    lean_history: List[str] = []  # Track Lean attempts and results
    answer_found = False
    lean_used = False  # Track if Lean has been used at least once
    first_call = True  # Track if this is the first LLM call

    try:
        for step_idx in range(args.max_steps):
            lean_state = "\n".join(lean_history[-6:]) if lean_history else "(no previous attempts)"

            decide_text, decide_raw = llm.decide(
                problem=problem,
                transcript="".join(transcript_blocks),
                lean_state=lean_state,
                lean_used=lean_used,
            )

            # Print system prompt on first call
            if first_call and hasattr(llm, 'messages') and len(llm.messages) >= 1:
                system_msg = llm.messages[0]
                print(f"[{system_msg.get('role', 'system')}]\n{system_msg.get('content', '')}\n")
                first_call = False

            if args.log_llm_io:
                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="llm_io",
                        t=time.time(),
                        payload={
                            "kind": "decide",
                            "output": decide_text,
                            "raw": decide_raw,
                        },
                    ),
                )

            kind, content = parse_decision(decide_text, require_lean=False)

            # Print assistant response
            print(f"[assistant]\n{decide_text}\n")

            # Handle ANSWER - final answer without full Lean verification
            if kind == "ANSWER":
                # if not lean_used:
                #     print(f"\n   ⚠️  Cannot ANSWER yet - must use Lean at least once first!")
                #     transcript_blocks.append(f"ATTEMPTED ANSWER (rejected - use Lean first): {content}\n")
                #     continue
                
                answer_found = True
                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="answer",
                        t=time.time(),
                        payload={"answer": content},
                    ),
                )
                transcript_blocks.append(f"ANSWER: {content}\n")
                break

            if kind == "THINK":
                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="think",
                        t=time.time(),
                        payload={"text": content},
                    ),
                )
                transcript_blocks.append(f"THINK: {content}\n")
                continue

            # LEAN code - this is an ACTION
            code = content
            lean_used = True  # Mark that Lean has been used

            append_jsonl(
                args.out,
                Event(
                    problem_id=args.problem_id,
                    event_type="lean_action",
                    t=time.time(),
                    payload={"step_idx": step_idx, "code": code},
                ),
            )

            success, result = lean_runner.run_code(code)

            append_jsonl(
                args.out,
                Event(
                    problem_id=args.problem_id,
                    event_type="lean_result",
                    t=time.time(),
                    payload={
                        "step_idx": step_idx,
                        "success": success,
                        "result": result,
                    },
                ),
            )

            if success:
                lean_history.append(f"ACTION:\n{code}\nOBSERVATION: ✓ Success")
            else:
                lean_history.append(f"ACTION:\n{code}\nOBSERVATION: ✗ Error: {result}")

            # Add observation to conversation for next turn
            llm.add_observation(code, result, success)
            
            # Print user message (Lean output)
            print(f"[user]\nLEAN OUTPUT:\n{result}\n")

            # Update transcript
            block = f"ACTION (Lean):\n{code}\nOBSERVATION: {'✓ Success: ' + result if success else '✗ ' + result}\n"
            transcript_blocks.append(block)

    except KeyboardInterrupt:
        print("\n\n[interrupted]")
    except Exception as e:
        append_jsonl(
            args.out,
            Event(
                problem_id=args.problem_id,
                event_type="crash",
                t=time.time(),
                payload={"error": repr(e)},
            ),
        )
        raise

    if args.pretty:
        with open(args.pretty, "w", encoding="utf-8") as f:
            # Write the full raw conversation
            if hasattr(llm, 'messages') and llm.messages:
                for msg in llm.messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    f.write(f"[{role}]\n{content}\n\n")


if __name__ == "__main__":
    main()
