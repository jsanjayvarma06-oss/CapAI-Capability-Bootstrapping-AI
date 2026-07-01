"""
capai.skill_executor
======================
Takes the raw repo content fetched by skill_discoverer.py and uses
Groq to:
  1. Understand what skills the repo provides
  2. Identify which skill(s) are relevant to the user's request
  3. Apply/adapt the skill to produce the actual output

This is intentionally separate from code_writer.py — skill_executor
READS and APPLIES existing pre-built code/templates, whereas
code_writer GENERATES new code from scratch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import llm_client

_UNDERSTAND_PROMPT = """\
You are analysing a GitHub skill repository to understand what it provides.

REPOSITORY: {repo_name}
DESCRIPTION: {description}

README:
{readme}

FILES:
{files_summary}

Answer these questions briefly:
1. What is this repository about? (1 sentence)
2. What specific skills/templates/components does it provide? (bullet list)
3. What types of user requests would this repo be useful for? (bullet list)
4. Which files contain the most useful/reusable content?
"""

_APPLY_PROMPT = """\
You are a skilled developer. A user has made a request and you have access to \
skill repositories with pre-built code and templates. Use the relevant content \
from the repositories to fulfil the request.

USER REQUEST: {request}

AVAILABLE SKILL REPOSITORIES:
{repos_content}

Instructions:
- Read through the repository content carefully
- Identify the most relevant templates, components, or code patterns
- Adapt them to specifically address the user's request
- Return the complete, ready-to-use output (full HTML page, full component, etc.)
- If multiple repos are available, combine the best parts from each
- Do NOT explain what you're doing — just return the output directly
- If it's a web page/component, return complete working code the user can use immediately

Return ONLY the final output with no explanation or preamble.
"""


@dataclass
class SkillResult:
    success: bool
    output: str = ""
    repos_used: list = field(default_factory=list)
    error: str = ""


def _summarise_files(files: list[dict], max_chars: int = 8000) -> str:
    """Build a compact summary of all files for the LLM prompt."""
    parts = []
    total = 0
    for f in files:
        header = f"\n--- {f['path']} ---\n"
        content = f["content"]
        chunk = header + content
        if total + len(chunk) > max_chars:
            remaining = max_chars - total - len(header) - 50
            if remaining > 100:
                parts.append(header + content[:remaining] + "\n...[truncated]")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


def apply_skill(request: str, repo_contents: list[dict]) -> SkillResult:
    """
    Use Groq to understand the fetched skill repos and apply the most
    relevant content to the user's request. Returns the final output.
    """
    if not repo_contents:
        return SkillResult(success=False, error="No skill repositories found or fetched.")

    if not llm_client.config.LLM_ENABLED:
        return SkillResult(
            success=False,
            error="No LLM configured — skill execution requires Groq or Anthropic."
        )

    # build a combined repos content string for the apply prompt
    repos_text_parts = []
    repos_used = []

    for repo in repo_contents:
        repo_name = repo.get("repo_name", "unknown")
        readme    = repo.get("readme", "")
        files     = repo.get("files", [])
        desc      = repo.get("description", "")
        url       = repo.get("url", "")

        files_summary = _summarise_files(files, max_chars=6000)

        section = (
            f"=== REPO: {repo_name} ===\n"
            f"URL: {url}\n"
            f"Description: {desc}\n\n"
            f"README:\n{readme[:2000]}\n\n"
            f"FILES:\n{files_summary}\n"
        )
        repos_text_parts.append(section)
        repos_used.append({"repo": repo_name, "url": url, "files": len(files)})

    repos_content = "\n\n".join(repos_text_parts)

    # ask Groq to apply the skill
    try:
        output = llm_client.complete(
            _APPLY_PROMPT.format(request=request, repos_content=repos_content),
            max_tokens=3000,
        )
        # strip markdown fences if present
        match = re.match(r"^```(?:\w+)?\s*\n(.*?)\n```\s*$", output.strip(), re.DOTALL)
        if match:
            output = match.group(1)
        return SkillResult(success=True, output=output.strip(), repos_used=repos_used)
    except Exception as e:
        return SkillResult(success=False, error=str(e), repos_used=repos_used)
