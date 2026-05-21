"""ReAct agent loop for the CV-paper RAG chatbot.

Manual Thought->Act->Observe loop driven by Gemini's function-calling
mechanism. Each round the LLM picks a tool; we (Python) dispatch it,
append the observation to the conversation, and re-call the LLM until it
stops requesting tools (text-only response = final answer) or we hit
``max_steps``.

Compared to Gemini's Automatic Function Calling (used by the single-pass
pre-RAG flow in ``generator_gemini.run_pre_rag_pass``), the manual loop:

- Surfaces each (thought, action, observation) tuple to the UI trace.
- Short-circuits duplicate searches via :data:`tools.react_search_state`.
- Caps total LLM round-trips at ``max_steps`` to keep latency bounded.
- Forces a tools-disabled re-call if the model keeps asking for tools
  past the cap, so the user always gets a text answer.
"""

import datetime
from dataclasses import dataclass
from typing import Iterator, Union

from google.genai import types

from src.config import GEMINI_MODEL, REACT_MAX_STEPS
from src.rag import tools as rag_tools
from src.rag.generator_gemini import _get_client
from src.rag.tools import (
    get_react_tools,
    react_retrieval_state,
    react_search_state,
)


REACT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a research assistant for a question-answering system over "
    "arXiv computer vision papers. You have access to tools that search "
    "and inspect the indexed paper corpus. Your job is to gather relevant "
    "information by calling tools, then write a grounded final answer.\n"
    "\n"
    "Workflow per turn:\n"
    "1. Decide what information you need.\n"
    "2. Call one or more tools.\n"
    "3. Read the observations.\n"
    "4. Decide whether to call more tools or to write the final answer.\n"
    "5. To finish, output the final answer as plain text — do NOT call any "
    "tool. Whatever text you output is what the user sees.\n"
    "\n"
    "Rules:\n"
    "- Final answer must be grounded in the chunks retrieved by your tool "
    "calls. Do not use outside knowledge or invent details.\n"
    "- Vary your search queries — repeating the same query is detected "
    "and returns an empty observation.\n"
    "- You have a hard cap of {max_steps} tool-calling rounds; plan "
    "accordingly. Past the cap you will be forced to answer with whatever "
    "you have.\n"
    "- Cite paper titles in your answer when possible.\n"
    "\n"
    "Today's date is {today}."
)


@dataclass
class ReactStep:
    """One iteration of the ReAct loop where the LLM called a tool."""

    n: int
    thought: str   # text emitted alongside the function_call (often empty)
    action: str
    action_args: dict
    observation: str


@dataclass
class ReactFinish:
    """Terminal event: LLM produced a text answer (or we forced one)."""

    answer: str
    sources: list           # accumulated chunk dicts from search_papers calls
    terminated_by: str      # "natural" | "max_steps" | "error"


ReactEvent = Union[ReactStep, ReactFinish]


def _extract_function_calls(response) -> list:
    """Pull all function_call parts out of a Gemini response."""
    calls: list = []
    for cand in (getattr(response, "candidates", None) or []):
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                calls.append(fc)
    return calls


def _extract_text(response) -> str:
    """Pull concatenated text content from a Gemini response."""
    direct = getattr(response, "text", None)
    if direct:
        return direct
    pieces: list = []
    for cand in (getattr(response, "candidates", None) or []):
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            t = getattr(part, "text", None)
            if t:
                pieces.append(t)
    return "".join(pieces)


def _dispatch(tool_callables: dict, name: str, args: dict) -> str:
    """Look up and execute a tool callable, returning its string result."""
    fn = tool_callables.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}."
    try:
        result = fn(**args)
    except Exception as exc:  # noqa: BLE001
        return f"Error: tool {name} raised {type(exc).__name__}: {exc}"
    return str(result) if result is not None else ""


def run_react(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    max_steps: int = REACT_MAX_STEPS,
) -> Iterator[ReactEvent]:
    """Run a bounded ReAct loop for one user prompt; yield events as they happen.

    Yields one :class:`ReactStep` per tool call (in execution order), then
    exactly one :class:`ReactFinish` with the answer text and accumulated
    sources. Callers (UI or CLI) consume these to render the live trace
    plus the final answer.

    Args:
        prompt: The user's natural-language message.
        model: Gemini model name; defaults to ``GEMINI_MODEL`` from config.
        max_steps: Hard cap on tool-calling rounds before forcing an answer.
    """
    # Per-turn state reset so prior turns / prior modes don't leak in.
    react_retrieval_state.clear()
    react_search_state.clear()
    rag_tools.last_call_log.clear()

    tool_callables = {fn.__name__: fn for fn in get_react_tools()}

    try:
        client = _get_client()
    except RuntimeError as exc:
        yield ReactFinish(
            answer=f"[ReAct unavailable: {exc}]",
            sources=[],
            terminated_by="error",
        )
        return

    today = datetime.date.today().isoformat()
    system_instruction = REACT_SYSTEM_PROMPT_TEMPLATE.format(
        max_steps=max_steps, today=today
    )

    contents: list = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # tools= callables lets the SDK build schemas from signatures; AFC is
    # explicitly disabled so we own the dispatch loop.
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=get_react_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    )

    step = 0
    terminated_by = "natural"

    while step < max_steps:
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001
            yield ReactFinish(
                answer=f"[ReAct loop failed: {type(exc).__name__}: {exc}]",
                sources=list(react_retrieval_state.chunks),
                terminated_by="error",
            )
            return

        function_calls = _extract_function_calls(response)

        if not function_calls:
            # No tool requested -> the LLM is done; its text is the answer.
            answer = _extract_text(response).strip()
            if not answer:
                answer = "[no answer text produced]"
            yield ReactFinish(
                answer=answer,
                sources=list(react_retrieval_state.chunks),
                terminated_by="natural",
            )
            return

        # Append the model's full content (text + function_calls together) so
        # the next turn's prompt history accurately reflects what it said.
        first_cand = response.candidates[0] if response.candidates else None
        if first_cand and getattr(first_cand, "content", None):
            contents.append(first_cand.content)
        else:
            # Fallback: reconstruct from extracted parts (shouldn't happen).
            contents.append(types.Content(
                role="model",
                parts=[types.Part(function_call=fc) for fc in function_calls],
            ))

        thought = _extract_text(response).strip()

        # Dispatch each function call. One ReactStep yielded per call; build
        # the function_response Parts for the next user turn as we go.
        fr_parts: list = []
        for fc in function_calls:
            step += 1
            action = fc.name
            args = dict(fc.args or {})

            # Anti-loop guard: only meaningful for search_papers, since other
            # tools' duplicate calls are either harmless (set_time_range
            # repeated) or rare. Guard short-circuits a fresh retrieval and
            # nudges the LLM to try a different angle.
            if action == "search_papers":
                q = str(args.get("query", ""))
                if react_search_state.has_seen(q):
                    observation = (
                        f"[anti-loop guard] Already searched for {q!r} this "
                        "turn. Try a different angle or finalize your answer."
                    )
                else:
                    react_search_state.record(q)
                    observation = _dispatch(tool_callables, action, args)
            else:
                observation = _dispatch(tool_callables, action, args)

            fr_parts.append(types.Part.from_function_response(
                name=action, response={"result": observation}
            ))

            yield ReactStep(
                n=step,
                thought=thought,
                action=action,
                action_args=args,
                observation=observation,
            )
            # Only show thought once per response, even if response had
            # multiple parallel calls.
            thought = ""

            if step >= max_steps:
                terminated_by = "max_steps"
                break

        # Send all observations back to the LLM for its next decision (or for
        # the forced-final-answer call below if we hit the cap mid-batch).
        contents.append(types.Content(role="user", parts=fr_parts))

        if terminated_by == "max_steps":
            break

    # Forced-final-answer path: re-call WITHOUT tools so the model has to
    # produce text, even if it would have asked for another search.
    if terminated_by == "max_steps":
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=(
                "[system] You have used all available tool-calling rounds. "
                "Produce the final answer now using only the chunks already "
                "retrieved. Do not request more tools."
            ))],
        ))
        force_config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=force_config
            )
            answer = _extract_text(response).strip()
        except Exception as exc:  # noqa: BLE001
            answer = (
                f"[max-steps reached; forced final-answer call failed: "
                f"{type(exc).__name__}: {exc}]"
            )

        yield ReactFinish(
            answer=answer or "[no final answer produced]",
            sources=list(react_retrieval_state.chunks),
            terminated_by="max_steps",
        )
