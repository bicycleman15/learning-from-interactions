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

USAGE:
  python lean_sandbox.py \
    --provider openai --model gpt-5.2 \
    --problem "Convert (0,3) to polar coordinates." \
    --out traj.jsonl --pretty pretty.txt
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
class LeanRunner:
    """Run Lean 4 code using a pre-configured Mathlib project."""
    
    def __init__(self, timeout: int = 120, project_dir: str = None):
        self.timeout = timeout
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
                return True, result.stdout or "✓ Code compiled successfully!"
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
    def decide(self, *, problem: str, transcript: str, lean_state: str, policy_mode: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def comment(self, *, problem: str, transcript: str, code: str, result: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError


class OpenAIResponsesLLM(LLM):
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def _call(self, prompt: str, *, max_output_tokens: int = 512, temperature: float = 0.3) -> Tuple[str, Dict[str, Any]]:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        raw = {"id": getattr(resp, "id", None)}
        return text, raw

    def decide(self, *, problem: str, transcript: str, lean_state: str, policy_mode: str, lean_used: bool = False, has_thought: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        if policy_mode == "UNBLOCK":
            mode_block = (
                "UNBLOCK_MODE: Your last Lean action produced an error.\n"
                "Read the error carefully. It contains useful information!\n"
                "Output a corrected LEAN action based on what you learned.\n"
            )
            prompt = f"""
I am trying to collect interaction logs with Lean for finetuning a smaller agent model.
Your job is to generate these interactions logs. 
In order to have useful interaction logs, you should make slight errors in your "reasoning" or "interaction" with Lean, and then rectify them based on the observations (output logs) from Lean.

Your final goal is to find the answer (which goes in \\boxed{{}}) using Lean 4 as kind of a computational tool.

Try to have mistakes in between so that we diversity in interaction logs.

{mode_block}

PROBLEM:
{problem}

LEAN HISTORY:
{lean_state}

TRANSCRIPT:
{transcript}
"""
        elif not has_thought:
            # Very minimal prompt to force THINK output - analysis only, no solving
            prompt = f"""You are solving a math problem. Before using any tools, you must THINK about the problem first.

PROBLEM:
{problem}

YOUR TASK: Output THINK: followed by your analysis and PLAN for how to solve this.

IMPORTANT: Do NOT compute the final answer yet! Just:
1. Identify what type of problem this is
2. Identify the key quantities and relationships
3. Plan what Lean code you'll write to help you find the answer

Do NOT solve the problem in your head - you will use Lean to compute the answer.
"""
        else:
            answer_note = "" if lean_used else " (requires at least one LEAN action first)"
            mode_block = (
                "Choose your next action:\n"
                "- THINK: <reason about the problem, plan, interpret observations>\n"
                "- LEAN:\n```lean\n<Lean 4 code to execute>\n```\n"
                f"- ANSWER: \\boxed{{<your final answer>}}{answer_note}\n"
            )
            prompt = f"""You are a mathematical reasoning agent solving competition math problems.
Your goal is to find the answer (which goes in \\boxed{{}}) using Lean 4 as a computational tool.

IMPORTANT: Do NOT compute answers in your head. Use Lean for ALL arithmetic and computation.
Lean is your calculator - you plan and analyze, Lean computes.

HOW TO USE LEAN FOR COMPUTATION:
- `#eval (125 : Rat) / 9` - rational arithmetic
- `#eval (2.5 : Float) * 3.0` - float arithmetic
- `#eval Float.sqrt 2` - square roots
- `#check <expr>` - check types
- `import Mathlib` - for ℝ, ℂ, trig, etc.

{mode_block}

PROBLEM:
{problem}

LEAN HISTORY:
{lean_state}

TRANSCRIPT:
{transcript}
"""
        return self._call(prompt, max_output_tokens=4000, temperature=0.9)

    def comment(self, *, problem: str, transcript: str, code: str, result: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = f"""Based on Lean's output, briefly note what you learned:
OBSERVATION: <what does this tell you? how does it help or redirect your approach?>

LEAN CODE:
{code}

LEAN OUTPUT:
{result}
"""
        return self._call(prompt, max_output_tokens=200, temperature=0.3)


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
    """
    text = text.strip()
    
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
    
    # Check for ANSWER
    m = ANSWER_RE.match(text)
    if m:
        return "ANSWER", m.group(1).strip()
    
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
    ap.add_argument("--problem", required=True, help="Problem statement for the model to solve")
    ap.add_argument("--problem_id", default="problem_0001")
    ap.add_argument("--out", default="traj.jsonl")
    ap.add_argument("--pretty", default=None)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--log_llm_io", action="store_true")

    args = ap.parse_args()

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

    lean_runner = LeanRunner(timeout=args.timeout)

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
                "problem": args.problem,
                "max_steps": args.max_steps,
                "timeout": args.timeout,
                "mode": "sandbox",
            },
        ),
    )

    transcript_blocks: List[str] = []
    lean_history: List[str] = []  # Track Lean attempts and results
    blocked = False
    last_error: Optional[str] = None
    proof_found = False
    lean_used = False  # Track if Lean has been used at least once
    has_thought = False  # Track if model has done THINK at least once

    print(f"\n{'='*60}")
    print(f"🧮 LEAN SANDBOX")
    print(f"{'='*60}")
    print(f"\n📝 PROBLEM: {args.problem}\n")
    print(f"The model will decide what theorem to prove.\n")

    try:
        for step_idx in range(args.max_steps):
            print(f"\n{'─'*60}")
            print(f"Step {step_idx + 1}")
            print(f"{'─'*60}")

            policy_mode = "UNBLOCK" if blocked else "NORMAL"
            if blocked:
                print(f"\n⚠️  MODE: UNBLOCK (previous code had errors)")

            lean_state = "\n".join(lean_history[-6:]) if lean_history else "(no previous attempts)"
            
            # Debug: show transcript being sent
            if transcript_blocks:
                print(f"\n📜 Transcript so far:\n{''.join(transcript_blocks)}")

            decide_text, decide_raw = llm.decide(
                problem=args.problem,
                transcript="".join(transcript_blocks),
                lean_state=lean_state,
                policy_mode=policy_mode,
                lean_used=lean_used,
                has_thought=has_thought,
            )

            if args.log_llm_io:
                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="llm_io",
                        t=time.time(),
                        payload={
                            "kind": "decide",
                            "policy_mode": policy_mode,
                            "output": decide_text,
                            "raw": decide_raw,
                        },
                    ),
                )

            kind, content = parse_decision(decide_text, require_lean=blocked)

            print(f"\n🤖 AGENT ({kind}):")
            if kind == "THINK":
                print(f"   {content}")
            elif kind == "ANSWER":
                print(f"   {content}")
            else:
                # Print full Lean code without truncation
                print(f"\n{content}")

            # Handle ANSWER - final answer without full Lean verification
            if kind == "ANSWER":
                if not lean_used:
                    print(f"\n   ⚠️  Cannot ANSWER yet - must use Lean at least once first!")
                    transcript_blocks.append(f"ATTEMPTED ANSWER (rejected - use Lean first): {content}\n")
                    continue
                    
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
                print(f"\n✅ Final answer recorded.")
                print(f"\n{'='*60}")
                print(f"📝 ANSWER: {content}")
                print(f"{'='*60}")
                break

            if kind == "THINK":
                if blocked:
                    print("\n   (Ignoring THINK in UNBLOCK mode - need LEAN action)")
                    continue
                    
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
                print(f"\n💭 Reasoning recorded.")
                has_thought = True
                continue

            # LEAN code - this is an ACTION
            # Enforce THINK first
            if not has_thought:
                print(f"\n   ⚠️  Must THINK first before using LEAN!")
                transcript_blocks.append(f"SYSTEM: REJECTED - You tried to use LEAN without THINKing first. Your next output MUST start with 'THINK:' followed by your analysis of the problem.\n")
                continue
                
            code = content
            lean_used = True  # Mark that Lean has been used
            print(f"\n⚡ Executing Lean action...")

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
                print(f"\n✅ OBSERVATION: Lean compiled successfully!")
                print(f"   {result}")
                lean_history.append(f"ACTION:\n{code}\nOBSERVATION: ✓ Success")
            else:
                print(f"\n❌ OBSERVATION: Lean error")
                print(f"   {result}")
                lean_history.append(f"ACTION:\n{code}\nOBSERVATION: ✗ Error: {result}")

            # Get agent's interpretation of the observation
            observation_text, observation_raw = llm.comment(
                problem=args.problem,
                transcript="".join(transcript_blocks),
                code=code,
                result=result,
            )

            observation = parse_observation(observation_text)
            print(f"\n🔍 {observation}")

            append_jsonl(
                args.out,
                Event(
                    problem_id=args.problem_id,
                    event_type="observation",
                    t=time.time(),
                    payload={"step_idx": step_idx, "observation": observation},
                ),
            )

            # Update transcript
            block = f"ACTION (Lean):\n{code}\nOBSERVATION: {'✓ Success' if success else '✗ ' + result}\n{observation}\n"
            transcript_blocks.append(block)

            if success:
                proof_found = True
                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="finished",
                        t=time.time(),
                        payload={"step_idx": step_idx, "code": code},
                    ),
                )
                print(f"\n{'='*60}")
                print(f"🎉 PROOF COMPLETE!")
                print(f"{'='*60}")
                break

            # Error: enter unblock mode
            blocked = True
            last_error = result

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
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
            f.write(f"PROBLEM:\n{args.problem}\n\n")
            f.write("".join(transcript_blocks))
            if proof_found:
                f.write("\n✓ Proof found!\n")

    if not proof_found:
        print(f"\n⚠️  No valid proof found in {args.max_steps} steps.")


if __name__ == "__main__":
    main()
