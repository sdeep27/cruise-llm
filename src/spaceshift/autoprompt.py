"""
AutoPrompt mode — a prompting model autonomously iterates on a prompt via tools.

The prompter can mutate the system prompt, the user prompt, or preset followups
(using the LLM class's sys/user/queue scaffolding). After each mutation, both
the current best and the challenger are run through the output model, and a
pairwise evaluation decides the winner. The prompter is told only win/loss on
the raw outputs; the evaluation metrics themselves can be shown to or hidden
from the prompter via the `expose_metrics` flag (default: shown).
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .LLM import LLM
from .evaluate import _generate_metrics, pairwise_evaluate
from .utils import _write_md


@dataclass
class Candidate:
    system: str = ""
    user_prompt: str = ""
    followups: list = field(default_factory=list)
    output: str | None = None


AUTOPROMPT_SYS_BASE = """You are an autonomous prompt optimizer. You will be shown a candidate prompt and must improve it by calling exactly one tool per turn.

The candidate has three parts, all editable:
  - system: the system prompt sent to the output model (starts empty)
  - user: the original user question (must stay on-topic)
  - followups: preset user messages that run sequentially after the initial response

Available tools:
  - edit_user_prompt: replace the user question entirely
  - edit_system_prompt: replace the system prompt (persona, constraints, output format)
  - edit_followup: replace an existing followup at a 0-indexed position
  - add_followup: append a new followup at the end
  - remove_followup: delete a followup at a 0-indexed position

You will receive terse win/loss feedback after each turn — the raw outputs are
hidden. A WIN means your edit becomes the new baseline. A LOSS means your edit
is reverted and you retry from the prior baseline.

Diversify your edits. Editing only the user prompt is rarely optimal — the
system prompt can shape tone/format globally, and followups can force iterative
refinement (e.g. "now cite every claim" or "identify anything you missed").

You MUST call exactly one tool per turn. Keep rationales short (one sentence)."""


_METRICS_EXPOSED_SUFFIX = """

Evaluation metrics (the SAME three axes apply every turn — target your edits at improving these):

{metrics_numbered}

Avoid edits that sacrifice one axis to gain on another unless you have a strong reason. These metrics are fixed and will not change across turns."""


_METRICS_BLIND_SUFFIX = """

Evaluation metrics are hidden from you. You do not know what axes the judge uses — explore broadly across quality, structure, specificity, and format rather than guessing."""


def _build_autoprompt_sys(resolved_metrics: list, expose_metrics: bool) -> str:
    if expose_metrics and resolved_metrics:
        numbered = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(resolved_metrics))
        return AUTOPROMPT_SYS_BASE + _METRICS_EXPOSED_SUFFIX.format(metrics_numbered=numbered)
    return AUTOPROMPT_SYS_BASE + _METRICS_BLIND_SUFFIX


def _run_candidate(c: Candidate, output_model: str, enable_search: bool, v: bool) -> str:
    """Execute a candidate — returns the final assistant message."""
    llm = LLM(model=output_model, v=v, search=enable_search, stream=False)
    if c.system:
        llm.sys(c.system, append=False)
    llm.user(c.user_prompt)
    for f in c.followups:
        llm.queue(f)
    return llm.result()


def _apply_edit(current: Candidate, edit: dict) -> Candidate | None:
    """Build a new Candidate from an edit spec. Returns None if invalid."""
    op = edit.get("op")
    if op is None:
        return None
    new = Candidate(
        system=current.system,
        user_prompt=current.user_prompt,
        followups=list(current.followups),
    )
    if op == "edit_user_prompt":
        new.user_prompt = edit["new_prompt"]
    elif op == "edit_system_prompt":
        new.system = edit["new_prompt"]
    elif op == "edit_followup":
        pos = edit["position"]
        if not isinstance(pos, int) or pos < 0 or pos >= len(new.followups):
            return None
        new.followups[pos] = edit["new_prompt"]
    elif op == "add_followup":
        new.followups.append(edit["new_prompt"])
    elif op == "remove_followup":
        pos = edit["position"]
        if not isinstance(pos, int) or pos < 0 or pos >= len(new.followups):
            return None
        new.followups.pop(pos)
    else:
        return None
    return new


def _format_candidate(c: Candidate) -> str:
    """Render a candidate as a readable block for the prompter."""
    lines = []
    lines.append("  system:")
    lines.append(f"    {c.system!r}" if c.system else "    (empty)")
    lines.append("  user:")
    lines.append(f"    {c.user_prompt!r}")
    lines.append("  followups:")
    if not c.followups:
        lines.append("    (none)")
    else:
        for i, f in enumerate(c.followups):
            lines.append(f"    [{i}] {f!r}")
    return "\n".join(lines)


def _build_context_msg(current: Candidate, feedback: str, turn: int, max_turns: int) -> str:
    return (
        f"Turn {turn}/{max_turns}\n\n"
        f"Feedback from last turn: {feedback}\n\n"
        f"Current baseline candidate:\n{_format_candidate(current)}\n\n"
        f"Call exactly one tool to mutate this candidate."
    )


def _save_autoprompt(result: dict, save_dir: str) -> list[str]:
    import os as _os
    paths = []
    base = _os.path.join(save_dir, "autoprompt")
    for entry in result["history"]:
        turn = entry["turn"]
        fname = f"turn_{turn:02d}.md"
        meta = {
            "turn": turn,
            "status": entry.get("status", entry.get("winner", "")),
        }
        if "edit" in entry and entry["edit"]:
            edit = entry["edit"]
            meta["edit_op"] = edit.get("op") or ""
            if "rationale" in edit:
                meta["rationale"] = edit["rationale"]
            if "position" in edit:
                meta["position"] = edit["position"]
        if "scores" in entry:
            scores = entry["scores"]
            meta["score_current"] = round(float(scores.get(0, 0.0)), 3)
            meta["score_challenger"] = round(float(scores.get(1, 0.0)), 3)
        cand = entry["candidate"]
        body = []
        body.append(f"# Turn {turn}\n")
        body.append(f"**Status:** {meta.get('status', '')}\n")
        if "edit_op" in meta:
            body.append(f"**Edit:** `{meta['edit_op']}`  ")
            if "rationale" in meta:
                body.append(f"**Rationale:** {meta['rationale']}\n")
        body.append("\n## System\n")
        body.append(f"```\n{cand['system']}\n```" if cand['system'] else "*(empty)*")
        body.append("\n## User\n")
        body.append(f"```\n{cand['user_prompt']}\n```")
        body.append("\n## Followups\n")
        if cand['followups']:
            for i, f in enumerate(cand['followups']):
                body.append(f"**[{i}]** {f}\n")
        else:
            body.append("*(none)*\n")
        body.append("\n## Output\n")
        body.append(cand.get('output') or "*(not generated)*")
        paths.append(_write_md(_os.path.join(base, fname), "\n".join(body), meta))

    # summary
    summary_lines = []
    summary_lines.append(f"# AutoPrompt Summary\n")
    summary_lines.append(f"**Original prompt:** {result['prompt']}\n")
    cfg = result.get("configs", {})
    summary_lines.append(f"**Prompting model:** {cfg.get('prompt_model')}  ")
    summary_lines.append(f"**Output model:** {cfg.get('output_model')}  ")
    summary_lines.append(f"**Evaluation model:** {cfg.get('eval_model')}  ")
    summary_lines.append(f"**Max turns:** {cfg.get('max_turns')}  ")
    summary_lines.append(f"**Search enabled:** {cfg.get('enable_search')}  ")
    summary_lines.append(f"**Metrics exposed to prompter:** {cfg.get('expose_metrics', False)}\n")
    resolved = result.get("resolved_metrics") or []
    if resolved:
        summary_lines.append("## Evaluation metrics\n")
        for i, m in enumerate(resolved):
            summary_lines.append(f"{i+1}. {m}")
        summary_lines.append("")
    summary_lines.append("## Turn-by-turn\n")
    summary_lines.append("| Turn | Edit | Status | Rationale |")
    summary_lines.append("|------|------|--------|-----------|")
    for entry in result["history"]:
        edit = entry.get("edit") or {}
        op = edit.get("op", "-")
        rationale = (edit.get("rationale") or "").replace("|", "\\|").replace("\n", " ")
        status = entry.get("status", entry.get("winner", ""))
        summary_lines.append(f"| {entry['turn']} | {op} | {status} | {rationale} |")
    summary_lines.append("\n## Final candidate\n")
    fc = result["final_candidate"]
    summary_lines.append("**System:**")
    summary_lines.append(f"```\n{fc['system']}\n```" if fc['system'] else "*(empty)*")
    summary_lines.append("\n**User:**")
    summary_lines.append(f"```\n{fc['user_prompt']}\n```")
    summary_lines.append("\n**Followups:**")
    if fc['followups']:
        for i, f in enumerate(fc['followups']):
            summary_lines.append(f"- [{i}] {f}")
    else:
        summary_lines.append("*(none)*")
    summary_lines.append("\n## Final output\n")
    summary_lines.append(result.get("final_output") or "*(none)*")
    meta_summary = {
        "prompt": result["prompt"],
        "prompt_model": cfg.get("prompt_model"),
        "output_model": cfg.get("output_model"),
        "eval_model": cfg.get("eval_model"),
        "max_turns": cfg.get("max_turns"),
        "enable_search": cfg.get("enable_search"),
        "expose_metrics": cfg.get("expose_metrics", False),
        "metrics_auto_generated": cfg.get("metrics_auto_generated", False),
        "resolved_metrics": resolved,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    paths.append(_write_md(_os.path.join(base, "summary.md"), "\n".join(summary_lines), meta_summary))
    return paths


def run_autoprompt(
    prompt: str,
    prompt_model: str,
    output_model: str,
    eval_model: str,
    max_turns: int = 6,
    metrics: list | None = None,
    expose_metrics: bool = True,
    enable_search: bool = False,
    save_dir: str | None = None,
    on_event=None,
    v: bool = True,
) -> dict:
    """Autonomously iterate on a prompt via tool-driven mutations and pairwise eval.

    Args:
        prompt: Initial user question.
        prompt_model: Model used to propose edits (gets only win/loss feedback).
        output_model: Model that generates candidate responses.
        eval_model: Model used as judge in pairwise evaluation.
        max_turns: Number of optimization turns after the baseline.
        metrics: Custom metric questions, or None to auto-generate (once, upfront).
        expose_metrics: If True, the resolved metrics are shown in the prompter's
            system prompt so it can target them. If False, the prompter is blind.
        enable_search: Enable web search on the output model.
        save_dir: Optional directory to write per-turn markdown + summary.
        on_event: Optional callback fn(event: dict) for live CLI updates.
        v: Verbose for the output model.

    Returns:
        dict with prompt, final_candidate, final_output, history, resolved_metrics, configs, saved.
    """
    pending_edit: dict = {}

    def _emit(event: dict):
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                pass

    # Resolve metrics once, upfront, before the loop. _generate_metrics is
    # prompt-driven (not response-driven), so we can call it before any
    # output exists. This fixes the moving-yardstick bug where pairwise_evaluate
    # would otherwise re-roll the metric set every turn.
    if metrics is None:
        resolved_metrics = _generate_metrics(
            items=[],
            additional_information=None,
            prompts=[prompt],
            results=None,
            mode="pairwise",
            eval_model=eval_model,
        )
        auto_generated = True
    else:
        resolved_metrics = list(metrics)
        auto_generated = False

    _emit({
        "type": "metrics_resolved",
        "metrics": list(resolved_metrics),
        "auto_generated": auto_generated,
        "exposed": bool(expose_metrics),
    })

    def edit_user_prompt(new_prompt: str, rationale: str) -> str:
        """Replace the original user question sent to the output model. Use this to reword, narrow, broaden, or reframe the user's request directly."""
        pending_edit.clear()
        pending_edit.update(op="edit_user_prompt", new_prompt=new_prompt, rationale=rationale)
        return f"Queued: user prompt will be replaced. Rationale: {rationale}"

    def edit_system_prompt(new_prompt: str, rationale: str) -> str:
        """Replace the system prompt (upstream context/persona). Use this to shape how the output model approaches every turn — tone, expertise, constraints, output format. The system prompt starts empty."""
        pending_edit.clear()
        pending_edit.update(op="edit_system_prompt", new_prompt=new_prompt, rationale=rationale)
        return f"Queued: system prompt will be replaced. Rationale: {rationale}"

    def edit_followup(position: int, new_prompt: str, rationale: str) -> str:
        """Replace an existing followup at a 0-indexed position. Followups run after the initial response — use to refine, expand, or correct. Fails if position is out of range."""
        pending_edit.clear()
        pending_edit.update(op="edit_followup", position=position, new_prompt=new_prompt, rationale=rationale)
        return f"Queued: followup[{position}] will be replaced. Rationale: {rationale}"

    def add_followup(new_prompt: str, rationale: str) -> str:
        """Append a new followup message at the end. Each followup runs as another user turn after the prior assistant response — use for multi-step reasoning scaffolds or progressive refinement."""
        pending_edit.clear()
        pending_edit.update(op="add_followup", new_prompt=new_prompt, rationale=rationale)
        return f"Queued: new followup will be appended. Rationale: {rationale}"

    def remove_followup(position: int, rationale: str) -> str:
        """Delete the followup at a 0-indexed position. Use when a prior followup is hurting more than helping. Fails if position is out of range."""
        pending_edit.clear()
        pending_edit.update(op="remove_followup", position=position, rationale=rationale)
        return f"Queued: followup[{position}] will be removed. Rationale: {rationale}"

    autoprompt_sys = _build_autoprompt_sys(resolved_metrics, expose_metrics)
    prompter = LLM(model=prompt_model, v=False, stream=False).sys(autoprompt_sys, append=False)
    prompter.tools(
        fns=[edit_user_prompt, edit_system_prompt, edit_followup, add_followup, remove_followup],
        tool_choice="required",
        parallel_tool_calls=False,
        max_turns=1,
    )

    current = Candidate(user_prompt=prompt)
    _emit({"type": "baseline_start"})
    current.output = _run_candidate(current, output_model, enable_search, v)
    _emit({"type": "baseline_done", "output": current.output})

    history = [{
        "turn": 0,
        "status": "baseline",
        "edit": None,
        "candidate": asdict(current),
    }]

    feedback = "No edits yet — this is the baseline. Propose your first change."

    for turn in range(1, max_turns + 1):
        pending_edit.clear()
        _emit({"type": "turn_start", "turn": turn, "max_turns": max_turns})

        try:
            prompter.user(_build_context_msg(current, feedback, turn, max_turns)).chat()
        except Exception as e:
            _emit({"type": "prompter_error", "turn": turn, "error": str(e)})
            feedback = f"Your last turn errored at the prompting model: {e}. Try a simpler edit."
            history.append({"turn": turn, "status": "prompter_error", "edit": None, "candidate": asdict(current), "error": str(e)})
            continue

        edit_snapshot = dict(pending_edit) if pending_edit else None
        _emit({"type": "edit_proposed", "turn": turn, "edit": edit_snapshot})

        challenger = _apply_edit(current, pending_edit) if pending_edit else None
        if challenger is None:
            feedback = "Your last tool call was invalid (unknown op or out-of-range position). Try a different edit."
            history.append({"turn": turn, "status": "invalid_edit", "edit": edit_snapshot, "candidate": asdict(current)})
            _emit({"type": "invalid_edit", "turn": turn, "edit": edit_snapshot})
            continue

        _emit({"type": "challenger_start", "turn": turn, "candidate": asdict(challenger)})
        try:
            challenger.output = _run_candidate(challenger, output_model, enable_search, v)
        except Exception as e:
            feedback = f"Running the challenger candidate failed: {e}. Your edit may have produced an invalid configuration. Try something simpler."
            history.append({"turn": turn, "status": "challenger_error", "edit": edit_snapshot, "candidate": asdict(challenger), "error": str(e)})
            _emit({"type": "challenger_error", "turn": turn, "error": str(e)})
            continue
        _emit({"type": "challenger_done", "turn": turn, "output": challenger.output})

        _emit({"type": "eval_start", "turn": turn})
        try:
            eval_result = pairwise_evaluate(
                prompt=prompt,
                results=[current.output, challenger.output],
                metrics=resolved_metrics,
                eval_model=eval_model,
                position_swap=True,
                v=False,
            )
        except Exception as e:
            feedback = f"Pairwise evaluation failed: {e}. Keeping prior baseline."
            history.append({"turn": turn, "status": "eval_error", "edit": edit_snapshot, "candidate": asdict(challenger), "error": str(e)})
            _emit({"type": "eval_error", "turn": turn, "error": str(e)})
            continue

        scores = eval_result["scores"]
        winner_is_challenger = eval_result["rankings"][0] == 1
        status = "win" if winner_is_challenger else "loss"
        _emit({
            "type": "eval_done",
            "turn": turn,
            "status": status,
            "scores": scores,
            "edit": edit_snapshot,
        })

        history.append({
            "turn": turn,
            "status": status,
            "winner": "challenger" if winner_is_challenger else "current",
            "edit": edit_snapshot,
            "candidate": asdict(challenger),
            "scores": scores,
        })

        if winner_is_challenger:
            current = challenger
            feedback = (
                f"Turn {turn}: your edit WON (current {scores.get(0, 0):.2f} vs challenger {scores.get(1, 0):.2f}). "
                "Your new candidate is now the baseline."
            )
        else:
            feedback = (
                f"Turn {turn}: your edit LOST (current {scores.get(0, 0):.2f} vs challenger {scores.get(1, 0):.2f}). "
                "Reverting to the prior baseline. Try a different direction."
            )

    result = {
        "prompt": prompt,
        "final_candidate": asdict(current),
        "final_output": current.output,
        "history": history,
        "resolved_metrics": list(resolved_metrics),
        "configs": {
            "prompt_model": prompt_model,
            "output_model": output_model,
            "eval_model": eval_model,
            "max_turns": max_turns,
            "enable_search": enable_search,
            "metrics": metrics,
            "expose_metrics": expose_metrics,
            "metrics_auto_generated": auto_generated,
        },
        "saved": [],
    }

    if save_dir:
        result["saved"] = _save_autoprompt(result, save_dir)

    return result
