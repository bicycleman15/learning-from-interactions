#!/usr/bin/env python3
"""
lean_interact_collect.py

Collect interaction trajectories of the form:
- Narrative reasoning in natural language (NARRATION)
- Intermittent Lean probes with strict one-line ACTION
- Grounded STATE/RESULT from Lean (LeanDojo)
- COMMENT generated after seeing the real Lean RESULT
- If ACTION fails, force UNBLOCK mode (only ACTION) until fixed

INSTALL:
  pip install lean-dojo openai anthropic

ENV:
  export OPENAI_API_KEY="your_openai_api_key_here"
  export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

USAGE (OpenAI):
  python lean_interact_collect.py \
    --provider openai --model gpt-5.2 \
    --repo_url https://github.com/yangky11/lean4-example \
    --commit 7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f \
    --file_path Lean4Example.lean \
    --theorem hello_world \
    --problem "Convert (0,3) to polar coordinates." \
    --out traj.jsonl --pretty pretty.txt

USAGE (Anthropic):
  python lean_interact_collect.py \
    --provider anthropic --model <your-claude-opus-model-name> \
    --repo_url ... --commit ... --file_path ... --theorem ... \
    --problem "..." --out traj.jsonl --pretty pretty.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple, Union

from lean_dojo import LeanGitRepo, Theorem, Dojo
from lean_dojo.interaction.dojo import TacticState, ProofFinished, LeanError, ProofGivenUp


# -----------------------------
# Logging schema (JSONL events)
# -----------------------------
EventType = Literal[
    "meta",
    "narration",
    "lean_step",
    "comment",
    "llm_io",
    "finished",
    "given_up",
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
# LLM adapters
# -----------------------------
class LLM:
    def decide(self, *, problem: str, transcript: str, state_pp: str, policy_mode: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Return either 'NARRATION: ...' or 'ACTION: ...' (one-line)."""
        raise NotImplementedError

    def comment(self, *, problem: str, transcript: str, state_pp: str, action: str, result_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Return exactly one sentence prefixed with 'COMMENT:'."""
        raise NotImplementedError


class OpenAIResponsesLLM(LLM):
    def __init__(self, model: str):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI()
        self.model = model

    def _call(self, prompt: str, *, max_output_tokens: int = 256, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        # openai-python exposes resp.output_text helper
        text = (getattr(resp, "output_text", "") or "").strip()
        raw = {"id": getattr(resp, "id", None)}
        return text, raw

    def decide(self, *, problem: str, transcript: str, state_pp: str, policy_mode: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        mode_block = ""
        if policy_mode == "UNBLOCK":
            mode_block = (
                "UNBLOCK_MODE: Your last ACTION produced a Lean error.\n"
                "Output ONLY:\n"
                "ACTION: <one Lean command line>\n"
                "No narration.\n"
            )
        else:
            mode_block = (
                "NORMAL_MODE: You may output EITHER:\n"
                "- NARRATION: <natural language>\n"
                "- ACTION: <one Lean command line>\n"
                "If ACTION, it must be a single line. No backticks. No extra fields.\n"
            )

        prompt = (
            "You are a math-solving agent using Lean 4 as an interactive verifier.\n"
            "Do most reasoning in natural language. Use Lean to probe intermediate steps.\n"
            "Treat Lean feedback as navigation. If an action fails, fix the Lean blockage first.\n"
            "Prefer short Lean probes: simp?, ring, nlinarith, norm_num, have ... := by ..., rewriting.\n\n"
            f"{mode_block}\n"
            f"PROBLEM:\n{problem}\n\n"
            f"CURRENT_LEAN_STATE:\n{state_pp}\n\n"
            f"TRANSCRIPT_SO_FAR:\n{transcript}\n"
        )
        return self._call(prompt, max_output_tokens=160, temperature=0.3)

    def comment(self, *, problem: str, transcript: str, state_pp: str, action: str, result_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = (
            "You are a math-solving agent using Lean 4 as an interactive verifier.\n"
            "Given the Lean STATE, your ACTION, and Lean RESULT, output exactly ONE sentence:\n"
            "COMMENT: <one sentence>\n"
            "No other text.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"STATE:\n{state_pp}\n\n"
            f"ACTION:\n{action}\n\n"
            f"RESULT:\n{result_text}\n"
        )
        return self._call(prompt, max_output_tokens=80, temperature=0.2)


class AnthropicMessagesLLM(LLM):
    def __init__(self, model: str):
        from anthropic import Anthropic  # type: ignore
        self.client = Anthropic()
        self.model = model

    def _call(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # anthropic SDK typically returns list blocks in .content
        text = ""
        content = getattr(msg, "content", None)
        if isinstance(content, list) and content:
            # take concatenated text blocks if any
            parts = []
            for b in content:
                t = getattr(b, "text", None)
                if isinstance(t, str):
                    parts.append(t)
            text = "".join(parts).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = str(content).strip()

        raw = {"id": getattr(msg, "id", None)}
        return text, raw

    def decide(self, *, problem: str, transcript: str, state_pp: str, policy_mode: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        mode_block = ""
        if policy_mode == "UNBLOCK":
            mode_block = (
                "UNBLOCK_MODE: Your last ACTION produced a Lean error.\n"
                "Output ONLY:\n"
                "ACTION: <one Lean command line>\n"
                "No narration.\n"
            )
        else:
            mode_block = (
                "NORMAL_MODE: You may output EITHER:\n"
                "- NARRATION: <natural language>\n"
                "- ACTION: <one Lean command line>\n"
                "If ACTION, it must be a single line. No backticks. No extra fields.\n"
            )

        prompt = (
            "You are a math-solving agent using Lean 4 as an interactive verifier.\n"
            "Do most reasoning in natural language. Use Lean to probe intermediate steps.\n"
            "Treat Lean feedback as navigation. If an action fails, fix the Lean blockage first.\n"
            "Prefer short Lean probes: simp?, ring, nlinarith, norm_num, have ... := by ..., rewriting.\n\n"
            f"{mode_block}\n"
            f"PROBLEM:\n{problem}\n\n"
            f"CURRENT_LEAN_STATE:\n{state_pp}\n\n"
            f"TRANSCRIPT_SO_FAR:\n{transcript}\n"
        )
        return self._call(prompt, max_tokens=160, temperature=0.3)

    def comment(self, *, problem: str, transcript: str, state_pp: str, action: str, result_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = (
            "You are a math-solving agent using Lean 4 as an interactive verifier.\n"
            "Given the Lean STATE, your ACTION, and Lean RESULT, output exactly ONE sentence:\n"
            "COMMENT: <one sentence>\n"
            "No other text.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"STATE:\n{state_pp}\n\n"
            f"ACTION:\n{action}\n\n"
            f"RESULT:\n{result_text}\n"
        )
        return self._call(prompt, max_tokens=80, temperature=0.2)


# -----------------------------
# Parsing + guards
# -----------------------------
ACTION_RE = re.compile(r"^\s*ACTION:\s*(.+?)\s*$", re.DOTALL)
NARR_RE = re.compile(r"^\s*NARRATION:\s*(.+?)\s*$", re.DOTALL)
COMMENT_RE = re.compile(r"^\s*COMMENT:\s*(.+?)\s*$", re.DOTALL)


def first_line(s: str) -> str:
    return s.splitlines()[0].strip() if s else ""


def parse_decision(text: str, *, require_action: bool) -> Tuple[str, str]:
    """
    Returns (kind, content)
    kind in {"ACTION","NARRATION"}
    """
    text = text.strip()
    if require_action:
        m = ACTION_RE.match(text)
        if not m:
            # try to salvage: if model forgot prefix, treat first line as action
            candidate = first_line(text)
            if candidate:
                return "ACTION", candidate
            raise ValueError("Expected ACTION, got empty/invalid output.")
        return "ACTION", first_line(m.group(1))

    # normal mode: allow narration or action
    m = ACTION_RE.match(text)
    if m:
        return "ACTION", first_line(m.group(1))
    m = NARR_RE.match(text)
    if m:
        # keep narration as-is
        return "NARRATION", "NARRATION: " + m.group(1).strip()
    # salvage: if it looks like a tactic-ish line, treat as ACTION; else narration
    candidate = first_line(text)
    if looks_like_lean_command(candidate):
        return "ACTION", candidate
    return "NARRATION", "NARRATION: " + text


def parse_comment(text: str) -> str:
    text = text.strip()
    m = COMMENT_RE.match(text)
    if m:
        # single sentence preference; but we won't enforce hard beyond stripping to first line.
        return "COMMENT: " + first_line(m.group(1))
    # salvage: first line as comment
    return "COMMENT: " + first_line(text) if text else "COMMENT: (no comment)"


def looks_like_lean_command(line: str) -> bool:
    # heuristic: common tactics / commands
    toks = ["simp", "simp?", "ring", "nlinarith", "linarith", "norm_num", "intro", "exact", "apply", "rw", "have", "refine", "cases", "constructor"]
    return any(line.startswith(t) for t in toks)


def render_step_block(state_pp: str, action: str, result_text: str, comment: str) -> str:
    return f"STATE: {state_pp}\nACTION: {action}\nRESULT: {result_text}\n{comment}\n"


# -----------------------------
# Main rollout
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", required=False, help="Model name. (OpenAI default: gpt-5.2; Anthropic: REQUIRED)")
    ap.add_argument("--repo_url", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--file_path", required=True)
    ap.add_argument("--theorem", required=True, help="Theorem name within file")
    ap.add_argument("--problem", required=True, help="Problem statement text shown to the model")
    ap.add_argument("--problem_id", default="problem_0001")
    ap.add_argument("--out", default="traj.jsonl")
    ap.add_argument("--pretty", default=None, help="Optional pretty transcript output path")
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--additional_import", action="append", default=[], help="Repeatable; passed to Dojo(additional_imports=...)")
    ap.add_argument("--log_llm_io", action="store_true", help="Also log full prompts + raw responses (can be large).")

    args = ap.parse_args()

    model = args.model
    if args.provider == "openai" and not model:
        model = "gpt-5.2"
    if args.provider == "anthropic" and not model:
        raise SystemExit("For --provider anthropic you must pass --model (your Claude Opus model name).")

    llm: LLM
    if args.provider == "openai":
        llm = OpenAIResponsesLLM(model=model)
    else:
        llm = AnthropicMessagesLLM(model=model)

    # meta header
    append_jsonl(
        args.out,
        Event(
            problem_id=args.problem_id,
            event_type="meta",
            t=time.time(),
            payload={
                "provider": args.provider,
                "model": model,
                "repo_url": args.repo_url,
                "commit": args.commit,
                "file_path": args.file_path,
                "theorem": args.theorem,
                "problem": args.problem,
                "max_steps": args.max_steps,
                "timeout": args.timeout,
                "additional_imports": args.additional_import,
            },
        ),
    )

    repo = LeanGitRepo(args.repo_url, args.commit)
    thm = Theorem(repo, args.file_path, args.theorem)

    transcript_blocks: List[str] = []
    blocked = False
    last_error: Optional[str] = None

    try:
        with Dojo(thm, timeout=args.timeout, additional_imports=args.additional_import) as (dojo, state):
            assert isinstance(state, TacticState)
            print(f"\n{'='*60}")
            print(f"Starting proof of: {args.theorem}")
            print(f"Problem: {args.problem}")
            print(f"{'='*60}\n")
            for step_idx in range(args.max_steps):
                state_pp = state.pp

                print(f"\n{'─'*60}")
                print(f"Step {step_idx + 1}")
                print(f"{'─'*60}")
                print(f"\n📋 CURRENT STATE:\n{state_pp}")

                policy_mode = "UNBLOCK" if blocked else "NORMAL"
                if blocked:
                    print(f"\n⚠️  MODE: UNBLOCK (previous action failed)")
                    print(f"   Last error: {last_error}")
                # Decide: narration or action
                decide_text, decide_raw = llm.decide(
                    problem=args.problem,
                    transcript="".join(transcript_blocks),
                    state_pp=state_pp + (f"\n\nLAST_ERROR:\n{last_error}" if blocked and last_error else ""),
                    policy_mode=policy_mode,
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
                                "state_pp": state_pp,
                                "last_error": last_error,
                                "output": decide_text,
                                "raw": decide_raw,
                            },
                        ),
                    )

                kind, content = parse_decision(decide_text, require_action=blocked)

                print(f"\n🤖 LLM OUTPUT ({kind}):")
                print(f"   {content}")

                if kind == "NARRATION":
                    # guard: if blocked, ignore narration and keep requesting ACTION
                    if blocked:
                        continue
                    append_jsonl(
                        args.out,
                        Event(
                            problem_id=args.problem_id,
                            event_type="narration",
                            t=time.time(),
                            payload={"text": content},
                        ),
                    )
                    transcript_blocks.append(content + "\n")
                    print(f"\n💭 Narration recorded, continuing...")
                    continue

                # ACTION
                action = content
                print(f"\n⚡ Executing Lean tactic: {action}")
                if "\n" in action:
                    action = first_line(action)

                # Run Lean
                result = dojo.run_tac(state, action)

                # Normalize result text for logging
                if isinstance(result, LeanError):
                    result_text = f"LeanError: {result.message}"
                    print(f"\n❌ LEAN ERROR:\n   {result.message}")
                elif isinstance(result, ProofFinished):
                    result_text = "ProofFinished"
                    print(f"\n✅ PROOF FINISHED!")
                elif isinstance(result, ProofGivenUp):
                    result_text = f"ProofGivenUp: {result.message}"
                    print(f"\n🏳️  PROOF GIVEN UP: {result.message}")
                else:
                    # TacticState
                    result_text = result.pp
                    print(f"\n✓ Tactic succeeded. New state:\n   {result_text}")

                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="lean_step",
                        t=time.time(),
                        payload={
                            "step_idx": step_idx,
                            "state": state_pp,
                            "action": action,
                            "result": result_text,
                            "outcome": (
                                "error" if isinstance(result, LeanError)
                                else "given_up" if isinstance(result, ProofGivenUp)
                                else "finished" if isinstance(result, ProofFinished)
                                else "ok"
                            ),
                        },
                    ),
                )

                # COMMENT (after seeing real RESULT)
                comment_text, comment_raw = llm.comment(
                    problem=args.problem,
                    transcript="".join(transcript_blocks),
                    state_pp=state_pp,
                    action=action,
                    result_text=result_text,
                )

                if args.log_llm_io:
                    append_jsonl(
                        args.out,
                        Event(
                            problem_id=args.problem_id,
                            event_type="llm_io",
                            t=time.time(),
                            payload={
                                "kind": "comment",
                                "state_pp": state_pp,
                                "action": action,
                                "result": result_text,
                                "output": comment_text,
                                "raw": comment_raw,
                            },
                        ),
                    )

                comment = parse_comment(comment_text)
                print(f"\n💬 {comment}")

                append_jsonl(
                    args.out,
                    Event(
                        problem_id=args.problem_id,
                        event_type="comment",
                        t=time.time(),
                        payload={"step_idx": step_idx, "comment": comment},
                    ),
                )

                # Add pretty block to transcript
                block = render_step_block(state_pp, action, result_text, comment)
                transcript_blocks.append(block + "\n")

                # Update blocked/state
                if isinstance(result, LeanError):
                    blocked = True
                    last_error = result.message
                    # state unchanged
                    continue

                if isinstance(result, ProofGivenUp):
                    append_jsonl(
                        args.out,
                        Event(
                            problem_id=args.problem_id,
                            event_type="given_up",
                            t=time.time(),
                            payload={"step_idx": step_idx, "message": result.message},
                        ),
                    )
                    break

                if isinstance(result, ProofFinished):
                    append_jsonl(
                        args.out,
                        Event(
                            problem_id=args.problem_id,
                            event_type="finished",
                            t=time.time(),
                            payload={"step_idx": step_idx},
                        ),
                    )
                    break

                # ok -> next state
                assert isinstance(result, TacticState)
                state = result
                blocked = False
                last_error = None

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


if __name__ == "__main__":
    main()
