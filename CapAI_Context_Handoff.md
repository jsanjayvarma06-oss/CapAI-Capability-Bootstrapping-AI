# CapAI — Full Context Handoff Document

This document contains the complete context of a long conversation about a project called CapAI. Paste this entire document into a new chat to continue with full continuity.

---

## 1. PROJECT ORIGIN AND EVOLUTION (in order)

The idea went through several distinct evolutions during the conversation. Each stage matters because later stages build on earlier reasoning:

1. **Initial idea:** "A self-learning AI which can write its own code and redeploy itself by improvising itself." This was identified as related to real research called "Recursive Self-Improvement," with examples cited: Darwin Gödel Machine, SICA, MIT's SEAL, Google DeepMind's AlphaEvolve.

2. **Refinement to uniqueness:** When asked "is it unique," the idea was narrowed to: the AI cannot search the web (example), so instead of stopping, it writes the code required to gain that capability, then rewrites itself to have it permanently. This was named **"Capability Bootstrapping."**

3. **Naming:** The project was named **CapAI**.

4. **Architecture v1 (single-codebase):** Originally imagined as one AI system with 4 modules: Task Runner + Capability Checker, Code Writer, Safety Sandbox, Capability Registry, running in a single loop: Task → Attempt → Fail → Gap Detected → Code Written → Sandbox Tested → Registry Stored → Retry → Succeed.

5. **Plugin reframe:** User clarified CapAI is itself "a plugin which can be installed into an AI model for adding capabilities" — i.e., not a standalone agent, but **installable middleware** that wraps ANY existing AI model (Claude, GPT, Gemini, custom models) and gives it self-expansion ability via `pip install capai`.

6. **Two operating modes defined:**
   - **Mode A — Developer install:** A developer wraps their AI system with CapAI; gaps are resolved automatically during use.
   - **Mode B — Silent failure interception:** CapAI sits between the model and the world, intercepting failures before the user ever sees an error, resolving them transparently.

7. **User uploaded a PDF ("CapAI_Research_Gaps.pdf")** identifying 4 specific unsolved research gaps in the original single-codebase design (see Section 3 below).

8. **User proposed the MCP-based architecture** as the solution to all 4 gaps: instead of writing new code into a shared codebase, CapAI spins up a dedicated **MCP (Model Context Protocol) server** for each new capability. Each MCP has its own diagnostic agent, its own testing agent, and its own GitHub repo/versioning. A single orchestrator agent runs everything. This was developed through a back-and-forth Q&A (see Section 4).

9. **Fine-tuning decision:** User has Google Colab access, advanced Python experience, and 1–2 months available. Decided to fine-tune **CodeT5+ (220M params)** using **LoRA** instead of relying solely on Claude API, with a **feedback loop** where every successful capability acquisition becomes a new training example (true recursive self-improvement).

10. **Teacher review:** Presented to teacher, who initially thought the idea was "simple and very common." This was addressed with comparison tables (vs LangChain, AutoGPT, SICA, Darwin Gödel Machine, etc.), a literature-review-backed novelty argument, and a flowchart/animation walkthrough using the example task "search the web for the latest AI news."

11. **Final deliverable:** A formal 17-page DOCX submission report was generated and given to the teacher (see Section 8).

---

## 2. CORE ONE-LINE DEFINITION (use this verbatim in any new chat)

> "CapAI is a plugin that installs into any AI model and gives it the ability to detect its own capability gaps, write the code to fill them, test it, and permanently expand itself — with zero human intervention."

Key clarifications established and important to preserve:
- CapAI is **NOT** training a new model from scratch.
- CapAI is **NOT** changing how a model is trained (not fine-tuning the base model's core knowledge either, at least not initially).
- CapAI **uses an existing trained model (e.g. Claude API) as a code-writing tool** — the model's weights don't change; CapAI's own **system-level capabilities** change.
- Analogy used: a carpenter who only has a hammer — when he needs a screwdriver, instead of waiting for a human to buy one, he makes one himself, tests it, and keeps it permanently. No human involved.
- Answer to "if models already have all these capabilities via chat, what's the use of CapAI?": there's a difference between a model **knowing how** to do something conceptually vs. a **deployed system actually having** that capability wired in and callable. CapAI automatically bridges that gap at runtime, in isolated/production environments where there's no human in the loop to "just ask Claude."
- Once user added their own fine-tuned model into the loop, the framing upgraded to: CapAI is also a step toward genuine recursive self-improvement, because the model that diagnoses/writes capabilities itself gets better as CapAI is used (feedback loop).

---

## 3. THE 4 IMPLEMENTATION-LEVEL RESEARCH GAPS (from user's uploaded PDF)

### Gap 1 — Identifying the Correct Missing Capability
**Problem:** When a task fails, there may be multiple overlapping causes (e.g., predicting customer churn from a CSV could fail due to missing ML model skill, missing data-cleaning, missing feature-engineering, or missing evaluation-metric skill). The Task Runner only sees the final exception, not the causal chain. Identical error signatures (e.g., a KeyError) can mean different underlying gaps. LLMs tend to produce confidently wrong diagnoses.
**Open question from the PDF:** How can an agent distinguish a skill gap from a data-quality problem when the output is simply wrong with no exception raised at all?
**Research directions in the PDF:** error-signature taxonomy, ablation probing, a diagnostic sub-agent, causal graph over task steps.
**CapAI's solution (developed by user + assistant):** A dedicated **Diagnostic Agent inside each MCP server** — a domain specialist that runs ablation probing (executing partial sub-tasks to isolate where the pipeline first breaks) and builds a causal graph, rather than guessing from the final error alone.

### Gap 2 — Capability Management & Registry Bloat
**Problem:** The registry grows without bound since every failure triggers a new code-writing cycle, leading to version clutter and redundant modules (e.g., qr_generator_v1, v2, _final, _updated, _fixed all doing the same thing).
**Open question from the PDF:** What is the right unit of identity for a capability — its name, its behaviour on a test suite, or its embedding in a shared semantic space?
**Research directions in the PDF:** embedding-based clustering, behavioural fingerprinting, champion/challenger model, semantic versioning policy.
**CapAI's solution:** Each capability lives in its **own isolated MCP server with its own GitHub repo** for clean versioning. A **Manager Agent** monitors usage frequency, detects/merges duplicate MCPs, and retires unused ones — keeping the registry clean automatically.

### Gap 3 — Capability Verification Beyond Unit Tests
**Problem:** Unit tests only cover anticipated inputs; modules can pass sandbox tests but fail silently on real-world edge cases (example given: a discount calculator that stacks two discounts incorrectly without raising an exception).
**Open question from the PDF:** Can the Code Writer be prompted to simultaneously produce a module and a formal postcondition, enabling lightweight runtime assertion checking in production?
**Research directions in the PDF:** property-based testing, mutation testing, adversarial critic agent, specification synthesis.
**CapAI's solution:** A **Testing Agent inside each MCP** runs **3-layer verification**: Layer 1 — search online for real test cases (Stack Overflow, GitHub issues, domain benchmarks); Layer 2 — reuse test cases from MCP history if the same problem was faced before; Layer 3 — self-generate edge case tests if nothing is found. All layers must pass before integration.

### Gap 4 — Cross-Agent Capability Sharing
**Problem:** Each CapAI instance has its own private registry; multiple deployed agents can't share learned skills, so each must rediscover capabilities independently (example: Agent A builds OCR, Agent B needs OCR and must rebuild from scratch).
**Open question from the PDF:** What is the minimum metadata a capability must carry so a receiving agent can make an informed, automated trust decision without human review?
**Research directions in the PDF:** capability passport (signed metadata: author agent, test suite, pass rate, input schema, known limitations), federated registry, trust tiers, differential sandboxing.
**CapAI's solution:** A **GraphQL communication layer** lets any MCP query any other MCP's capabilities. **Trust tiers** govern usage: self-generated = full trust (use immediately); peer-shared (same network) = medium trust (re-run own sandbox tests before adopting); external/unverified = zero trust (blocked entirely).

---

## 4. FULL MCP-BASED ARCHITECTURE (final version)

### The 5 Agents

1. **Orchestrator Agent** (single instance — the brain)
   - Receives every task from user/external system
   - Checks Capability Registry first — executes immediately if capability exists
   - Detects "I cannot produce output" condition (the trigger signal) when a capability is missing
   - Triggers creation of a new MCP server
   - Retries the original task once the new capability is verified and live
   - Coordinates all other agents via GraphQL

2. **Diagnostic Agent** (one per MCP server — root cause specialist)
   - Domain specialist; analyses full error trace
   - Runs ablation probing (partial sub-tasks) to isolate exact failure point
   - Builds a causal graph (DAG) of task steps to find which node first produces invalid output
   - Handles compound failures (multiple overlapping causes)
   - Outputs a structured **Gap Report**: exact capability missing, input/output schema, constraints — never a guess

3. **Code Writer** (one per MCP server — LLM interface)
   - Takes the Gap Report and constructs a precise prompt to an LLM (Claude API initially; fine-tuned CodeT5+ model later)
   - Receives a working Python module
   - Saves it to the MCP's own dedicated GitHub repo with automatic version tagging (starts at v0.0.1)

4. **Testing Agent** (one per MCP server — 3-layer verifier)
   - Layer 1: online test cases (Stack Overflow, GitHub issues, domain benchmarks)
   - Layer 2: MCP history (reuse if a similar problem was solved before)
   - Layer 3: self-generated edge case tests if nothing found
   - Blocks integration on any failure; logs all results with timestamps

5. **Manager Agent** (single instance — resource & lifecycle controller)
   - Monitors all active MCPs and usage frequency
   - Detects and merges/removes duplicate MCPs
   - Decides lifecycle: keep, archive, or retire each MCP
   - Reviews and approves/rejects admission of a verified MCP into the **Main Folder / Main Registry**
   - Controls trust tiers for GraphQL-shared capabilities
   - Handles all file/code/resource issues across any MCP

### The MCP Server (one per capability)
- Fully isolated environment — one broken MCP cannot affect others
- Own GitHub repo (full history, rollback, clean versioning)
- Own Diagnostic Agent + own Testing Agent
- Exposes a GraphQL endpoint so other MCPs/agents can discover and query it
- Only joins the **Main Folder** (production registry) after passing Manager Agent review and full verification

### GraphQL Communication Layer
- All MCPs, the Orchestrator, and the Manager Agent communicate via GraphQL
- Benefits: query only what's needed, self-describing schema (auto-discovery of capabilities), single endpoint per MCP, cross-server compatibility, strongly typed, introspection support
- Example query structure: `capability(name: "ocr_module") { name version inputSchema outputSchema testPassRate lastVerified authorAgent knownLimitations }`
- Trust tiers: self-generated = full trust; peer MCP = medium trust (re-verify); external/unverified = zero trust (blocked)

### The 7 (sometimes described as 9/10) Stage Acquisition Loop
1. Task given to Orchestrator
2. Orchestrator checks registry — capability exists? If yes, execute instantly. If no, continue.
3. Orchestrator detects it cannot produce output (failure intercepted silently — user never sees raw error)
4. Manager Agent spins up a new MCP server (own environment, own GitHub repo, own Diagnostic + Testing agents)
5. Diagnostic Agent identifies exact root cause → Gap Report
6. Code Writer generates the capability module via LLM → saved to MCP repo v0.0.1
7. Testing Agent runs 3-layer verification — all layers must pass
8. Manager Agent reviews (checks duplicates/safety) and approves admission
9. MCP joins the Main Folder / Main Registry (tagged v1.0.0), now available to ALL agents via GraphQL
10. Orchestrator retries the original task — succeeds; capability is now permanent and reused instantly next time

### "CapAI is a plugin" — final framing
Two modes:
- **Mode A:** Developer installs CapAI (`pip install capai`) and wraps any base model: `CapAI(base_model="claude-sonnet-4-6")`, `gpt-4`, `gemini-pro`, or a custom fine-tuned model. Self-expansion then happens automatically during use.
- **Mode B:** CapAI silently intercepts failures at the system level — before the user ever sees an error — resolves the gap in the background (~45–60 seconds), and returns the correct result, fully transparently.

---

## 5. LITERATURE REVIEW — 10 PAPERS (2024–2025)

| # | Paper | Authors | Year | Source |
|---|-------|---------|------|--------|
| 1 | A Comprehensive Survey of Self-Evolving AI Agents | Fang et al. | 2025 | arXiv:2508.07407 |
| 2 | Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | Zhang et al. | 2025 | arXiv:2505.22954 |
| 3 | SICA: A Self-Improving Coding Agent | Robeyns, Szummer, Aitchison | 2025 | ICLR 2025 Workshop (OpenReview: rShJCyLsOr) |
| 4 | Large Language Models Can Self-Improve at Web Agent Tasks | Anonymous | 2024 | arXiv:2405.20309 |
| 5 | Live-SWE-Agent: Can Software Engineering Agents Self-Evolve on the Fly? | Xia et al. | 2025 | arXiv:2511.13646 |
| 6 | Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement | Yin et al. | 2024 | arXiv:2410.04444 |
| 7 | Lifelong Learning of Large Language Model Based Agents: A Roadmap | Anonymous | 2025 | IEEE Xplore 11328884 |
| 8 | Agentic AI: A Comprehensive Survey | Anonymous | 2025 | IEEE Xplore 11071266 |
| 9 | Large Language Models as Tool Makers (LATM) | Cai et al. | 2024 | arXiv:2305.17126 |
| 10 | CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning | Qian, Han, Fung, Qin, Liu, Ji | 2024 | arXiv:2305.14318 |

Other papers mentioned/considered but not in the final 10: Voyager (arXiv:2305.16291, Minecraft-only open-ended skill acquisition), Truly Self-Improving Agents Require Intrinsic Metacognitive Learning (ICML/OpenReview 4KhDd0Ozqe), Self-Improving AI Agents Through Self-Play (arXiv:2512.02731).

### 5 Research Gaps from the Literature (distinct from the 4 implementation gaps in Section 3)
1. **Optimisation vs. Capability Acquisition** — all reviewed systems (SICA, Darwin Gödel Machine, Gödel Agent) improve existing capabilities; none acquire entirely new ones from scratch.
2. **Absence of Failure-Driven Gap Detection** — failure is used for retry/refinement, not as a structured trigger for building a new capability.
3. **Human-Scoped Tool Creation** — LATM and CREATOR can create tools, but a human must define the task scope upfront; no autonomous gap detection.
4. **No Permanent, Shareable Capability Registry** — no reviewed system persists capabilities across sessions or shares them between agent instances.
5. **Absence of Independent Safety Verification** — systems that self-modify code (SICA, Darwin Gödel Machine) don't isolate new code in an independently verified sandbox before integration.

### Novel Contribution Statement (final, used in submitted report)
> "CapAI is the first installable plugin that gives any AI system the ability to autonomously detect its own capability gaps, write the code to fill them, test it, and permanently expand itself — with zero human intervention — a combination that does not exist in any published research."

### Comparative Table (CapAI vs. closest systems) — final version used in submission
| System | Failure-Driven | New Capability | Sandbox Verified | Permanent Registry | Universal Plugin |
|---|---|---|---|---|---|
| SICA (2025) | No | No | No | No | No |
| Darwin Gödel Machine (2025) | No | No | No | No | No |
| LATM (2024) | No | Partial | No | No | No |
| CREATOR (2024) | No | Partial | No | No | No |
| Live-SWE-Agent (2025) | Partial | Partial | No | No | No |
| **CapAI (proposed)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 6. FINE-TUNING PLAN

**User constraints:** Google Colab access, advanced Python/ML experience, 1–2 months available.

**Base model:** CodeT5+ (220M parameters, Salesforce) — chosen for existing code-pretraining and fitting Colab free-tier memory constraints.
**Fine-tuning method:** LoRA (Low-Rank Adaptation) — updates ~1% of parameters, fits in free Colab GPU memory.

**Dataset:** 1,000–2,000 (target ~1,500) training examples from:
- Stack Overflow API (real Python errors + accepted fixes)
- GitHub Issues API (bug reports + the PRs that fixed them)
- Synthetic generation via Claude API (error → gap → fix triples)

**Training example format:**
```
INPUT: task description + error/failure trace + context
OUTPUT: gap_label (e.g. "MISSING: WEB_SEARCH") + module_code + test_cases
```

**4-week plan:**
- Week 1 — Build dataset (collect, clean, dedupe, format as JSON; target 1500 examples)
- Week 2 — Fine-tune on Colab using LoRA on CodeT5+; checkpoints to Google Drive; ~45–60 min training on free T4 GPU
- Week 3 — Evaluate 3 configurations: (A) CapAI + Claude API baseline, (B) CapAI + fine-tuned model, (C) CapAI + fine-tuned model after 50 more real capability acquisitions via feedback loop. Metrics: gap detection accuracy, code correctness rate, response latency, cost.
- Week 4 — Plug fine-tuned model into the Code Writer in place of/alongside Claude API; implement the feedback loop in code.

**The Feedback Loop (key novelty):** Every time CapAI successfully adds a new capability, that example (error → gap → code → tests) is saved as a new training example. Once 50 new examples accumulate, the model is automatically re-fine-tuned. This means CapAI's own specialist model improves purely as a function of CapAI being used — genuine recursive self-improvement, distinct from the static optimization in all 10 reviewed papers.

**Expected results table (projected, not yet measured):**
| Metric | Claude API Baseline | Fine-Tuned (Week 3) | Fine-Tuned (Week 4+, feedback active) |
|---|---|---|---|
| Gap detection accuracy | ~70% | ~80% | ~88%+ |
| Code correctness rate | ~75% | ~80% | ~87%+ |
| Response time | 2–3 sec | 0.3–0.5 sec | 0.2–0.4 sec |
| Cost per capability | ~$0.01 | $0.00 | $0.00 |
| Improves over time | No | No (static) | Yes |
| Works offline | No | Yes | Yes |

---

## 7. PUBLICATION & PROTECTION ROADMAP

1. **arXiv (cs.AI)** — free preprint, immediate, establishes priority/timestamp. Do this first.
2. **USPTO Provisional Patent** — ~$320 for students, 1–3 months, protects the plugin architecture and acquisition loop specifically. (Discussed: AI-assisted inventions ARE patentable as of 2025 USPTO guidance as long as a human significantly contributed — confirmed applicable here since the team designed it.)
3. **NeurIPS / ICML Workshop** (e.g. Workshop on Autonomous Agents / Scaling Self-Improving Foundation Models) — 4–6 months, lighter review, good first peer-reviewed credit.
4. **Full Conference** (ICLR / NeurIPS / ICML main track) — 8–12 months, prestigious, long-term goal.
5. **Journal** (JAIR or IEEE Transactions on Neural Networks and Learning Systems) — 12–18 months, most rigorous, ultimate goal.
6. **Capstone option** — submit as university capstone project alongside the above (easiest, parallel track).

Suggested paper title: *"CapAI: Autonomous Capability Acquisition in AI Agents via Failure-Driven Code Generation"* (or the plugin-focused variant: *"CapAI: A Universal Self-Expanding Plugin for AI Systems — Autonomous Capability Acquisition via Failure Interception and MCP-Based Code Generation"*).

---

## 8. DELIVERABLES ALREADY PRODUCED IN THIS CONVERSATION

(Note: these files exist only in the previous chat's sandbox/output folder, NOT automatically available in a new chat — they would need to be re-generated or re-uploaded if needed again.)

1. **README.md** — for an earlier, unrelated project idea (Instagram GPU pre-warmer using engagement velocity as a predictive autoscaling signal) — this was the very first thing discussed before pivoting to CapAI.
2. **CapAI_README.md** — original project README with architecture, 4-week plan, resume pitch.
3. **CapAI_Paper_Reading_Template.pdf** — blank + filled-example template for reading research papers (3-round method: 5-min scan, 30-min read, deep dive).
4. **CapAI_Literature_Survey_Guide.pdf** — mapped teacher's exact guidelines (literature survey, table of papers, results/accuracy/latency/false-positives, future work from 2024+ papers, tools like SciSpace/Elicit/Research Rabbit/Connected Papers/Consensus, capstone/conference/journal/patent options, Google Scholar/IEEE Xplore) into a structured action plan.
5. **CapAI_Literature_Review.pdf** — formal literature review document with 5-part structure (papers list, research gaps, novel contribution, solution, conclusion) requested per teacher's instructions.
6. **CapAI_Team_Architecture_Guide.pdf** and **CapAI_Complete_Team_Guide.pdf** — detailed team-facing documents explaining the MCP architecture, all agents, GraphQL, gap solutions, fine-tuning plan, publication roadmap (the complete version was 11 parts).
7. **CapAI_Teacher_Explanation.pdf** — a simplified, jargon-free one-pager specifically designed to convince the skeptical teacher (carpenter analogy, 7-step delay table, comparison to ChatGPT/LangChain/AutoGPT/SICA, 3 pointed questions to ask back).
8. **CapAI_Flowchart_Example.pdf** — visual flowchart + step-by-step walkthrough using the example "search the web for AI news," including before/after timeline comparison and a 5-day capability growth table.
9. **CapAI_Animated_Flow.html** and **CapAI_Live_Animation.html** — interactive/animated HTML demos showing the full agent flow live with typing code, moving data packets, live agent status panels, test bars filling, and a final success state. The live animation version is the more cinematic, fully-featured one (5 agent cards, particle/packet canvas, live registry, system log, timer).
10. **CapAI_Blueprint.html** — a single-page technical "blueprint/schematic" style poster covering definition, system diagram, agent spec sheets, the acquisition loop, gap/solution index table, before/after comparison, and tech stack — designed as one comprehensive visual.
11. **CapAI_Architecture_Overview.png** — a clean, simple static architecture diagram (matplotlib-generated) showing User → Orchestrator → decision diamond (capability exists?) → MCP Server (Diagnostic Agent + Code Writer + Testing Agent) → GraphQL bus → Manager Agent → Main Registry → Task Retried/Success, with a feedback loop arrow back to the Orchestrator for instant reuse.
12. **CapAI_Submission_Report.docx** — the final formal 17-page submission document, with: title page, abstract, a STATIC table of contents with dot leaders and correct page numbers (the dynamic Word TOC field was initially broken/blank because it requires manual field update — this was fixed by replacing it with a static, manually-built TOC), Introduction, Literature Review (10-paper table + 5 gaps), Proposed System (with the architecture PNG embedded as Figure 1, all 5 agents detailed, the acquisition loop), Section 4 mapping the 4 implementation gaps to MCP solutions, Methodology (4-week plan + fine-tuning plan), Comparative Analysis table, Future Work, Publication Roadmap, Conclusion, References. This is the document actually given to the teacher.

**Tech stack used to build these:** Python (matplotlib for the PNG diagram), Node.js `docx` npm package (for the DOCX), LibreOffice headless (`soffice --headless --convert-to pdf`) used purely to QA/verify rendering before delivering files, HTML/CSS/vanilla JS for the animated demos.

---

## 9. OPEN QUESTIONS / UNRESOLVED ITEMS (carried over as Future Work)

1. How can CapAI distinguish a genuine skill gap from a data-quality problem when a task produces a wrong result with **no exception raised at all** (silent failure, not caught by current Diagnostic Agent design)?
2. What is the minimal "capability passport" metadata schema needed for a receiving MCP/agent to make an automated trust decision without human review?
3. Embedding-based semantic clustering for the Manager Agent (beyond current behavioural fingerprinting) to further reduce duplication.
4. Formal specification synthesis — having the Code Writer produce a machine-checkable postcondition alongside generated code.
5. Empirical validation across 3+ different base models (GPT-4, Claude, an open-weight model) to substantiate the "universal plugin" claim.

---

## 10. CURRENT STATUS AS OF END OF PREVIOUS CHAT

- **Nothing has been coded/implemented yet.** Everything so far is architecture design, literature review, documentation, and visualization.
- The formal DOCX report has been generated and given to the teacher; the TOC bug was found and fixed (replaced dynamic field with static dot-leader TOC).
- The most recently discussed but not-yet-started next step was: **begin actual Python implementation, starting with Week 1 — Task Runner + Capability Checker** (simple try/except-based failure capture and gap classification, no ML needed yet).
- User has Google Colab + advanced Python skills + 1–2 months for the fine-tuning portion specifically.
- No mentor/professor has been secured yet (was recommended, not yet done).
- No arXiv upload, patent filing, or workshop submission has happened yet — all roadmap items, not completed actions.

---

## 11. STYLE / PREFERENCE NOTES FOR THE NEW CHAT

- User responds well to: very simple analogies (carpenter/screwdriver, food stall/cricket match, VS Code plugins), step-by-step numbered plans, tables comparing old-vs-new or CapAI-vs-existing-systems, and visual/animated explanations for convincing skeptical audiences (teacher, teammates).
- User explicitly said at one point: "dont generate anything without my permission" — so in a new chat, **check before generating large files/documents** unless explicitly asked, though later requests in this same conversation directly asked for generation each time, effectively overriding that for explicit asks.
- User is working with a team (mentioned teammates needing to understand the architecture) and reports to a teacher for what appears to be a summer internship with literature-survey deliverables.
- User has iteratively pushed the idea to be more ambitious and technically deeper at almost every turn (plugin → universal plugin → MCP architecture → GraphQL → fine-tuning → feedback loop) — expect a similar appetite for going deeper rather than simplifying, when given the choice.

---

**End of context handoff. Paste this whole document at the start of a new chat and say "continue from this context" to resume seamlessly.**
