"""
capai.skill_discoverer
========================
Searches GitHub for skill repos relevant to a request, reads their
README + code files, and returns the raw content so skill_executor.py
can apply them. No GitHub auth needed for public repos.

Flow:
  1. Search GitHub for repos matching the request topic
  2. For each candidate repo, fetch README.md
  3. Fetch all .py / .html / .css / .js / .json files in root + skills/
  4. Return structured content for Groq to reason over
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Optional

GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
GITHUB_RAW        = "https://raw.githubusercontent.com"
GITHUB_API        = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "CapAI-SkillDiscoverer/1.0",
}

SKILL_CODE_EXTENSIONS = {".py", ".html", ".css", ".js", ".json", ".md", ".txt", ".yaml", ".yml"}
MAX_FILE_SIZE_BYTES   = 50_000   # skip files larger than 50KB to stay within LLM context
MAX_FILES_PER_REPO    = 15


def _gh_get(url: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _fetch_raw(url: str) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "CapAI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            if len(raw) > MAX_FILE_SIZE_BYTES:
                return f"[file too large — {len(raw)} bytes, truncated]\n" + raw[:MAX_FILE_SIZE_BYTES].decode("utf-8", errors="replace")
            return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def search_repos(query: str, max_repos: int = 3) -> list[dict]:
    """Search GitHub for repos matching the query. Returns list of repo metadata."""
    params = urllib.parse.urlencode({
        "q": f"{query} skill template",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos,
    })
    data = _gh_get(f"{GITHUB_SEARCH_API}?{params}")
    if not data or "items" not in data:
        return []
    return [
        {
            "full_name": r["full_name"],
            "description": r.get("description", ""),
            "stars": r.get("stargazers_count", 0),
            "url": r["html_url"],
            "default_branch": r.get("default_branch", "main"),
        }
        for r in data["items"]
    ]


def fetch_repo_contents(repo_full_name: str, branch: str = "main") -> dict:
    """
    Fetch README + all skill-relevant code files from a GitHub repo.
    Returns {readme, files: [{path, content}], repo_name}.
    """
    result = {"repo_name": repo_full_name, "readme": "", "files": []}

    # 1. fetch README
    for readme_name in ("README.md", "readme.md", "README.txt", "README"):
        raw_url = f"{GITHUB_RAW}/{repo_full_name}/{branch}/{readme_name}"
        content = _fetch_raw(raw_url)
        if content:
            result["readme"] = content
            break

    # 2. list files in root
    tree_data = _gh_get(f"{GITHUB_API}/repos/{repo_full_name}/git/trees/{branch}?recursive=1")
    if not tree_data or "tree" not in tree_data:
        return result

    # collect candidate files
    candidates = []
    for item in tree_data["tree"]:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext not in SKILL_CODE_EXTENSIONS:
            continue
        if path.lower() in ("readme.md", "readme.txt", "readme"):
            continue  # already fetched
        candidates.append(path)

    # fetch up to MAX_FILES_PER_REPO files
    for path in candidates[:MAX_FILES_PER_REPO]:
        raw_url = f"{GITHUB_RAW}/{repo_full_name}/{branch}/{path}"
        content = _fetch_raw(raw_url)
        if content:
            result["files"].append({"path": path, "content": content})

    return result


def discover_skill(request: str, repo_urls: Optional[list[str]] = None, max_repos: int = 3) -> list[dict]:
    """
    Main entry point. Given a plain-English request:
      - If repo_urls provided, fetch those specific repos directly.
      - Otherwise search GitHub for relevant skill repos.
    Returns list of repo content dicts for skill_executor to reason over.
    """
    repos_meta = []

    if repo_urls:
        for url in repo_urls:
            # extract owner/repo from URL
            parts = url.rstrip("/").replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                repos_meta.append({
                    "full_name": f"{parts[0]}/{parts[1]}",
                    "description": "",
                    "stars": 0,
                    "url": url,
                    "default_branch": "main",
                })
    else:
        repos_meta = search_repos(request, max_repos=max_repos)

    results = []
    for repo in repos_meta:
        contents = fetch_repo_contents(repo["full_name"], repo.get("default_branch", "main"))
        contents["stars"] = repo.get("stars", 0)
        contents["description"] = repo.get("description", "")
        contents["url"] = repo.get("url", "")
        results.append(contents)

    return results
