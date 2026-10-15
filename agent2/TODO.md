# TODO
### Planning
- [done] Pipeline composition: implemented as ACE worker graphs, see ../ACE_SCHEMA.md
- [done] Judging, merging, graph: JUDGEMERGER/PICKMERGER workers and NEAT-evolved graphs, see ../ACE_SCHEMA.md

### Coding, Core
- File view speculator

### Coding, Misc
- Tool headers/footers

### Follow-ups
- Multi-language codebase graph: register non-Python adapters and extract imports and references for them (ACE's graph is Python-only).
- SWE-bench scorer: a `module:function` scorer for ACE's TaskEvaluator that applies the Edit and runs the instance tests.
- Workspace.diff: keep line terminators and the no-newline marker so its output applies (ACE builds its own git patch).
- Session.refresh_files: avoid re-reading the project on every tool call, or pin a pristine snapshot for a run.
- replace_lines: reject a reversed line range instead of corrupting the file.
- attempt_fix_formatting: split on "\n" only, not str.splitlines (form feeds split lines today).
- DirectoryConstructor: substitute {{FILES*}} only in the configured prompt, not anywhere in the chat.
- make_empty_files: do not truncate files that already exist.
- CodeFile.code_nodes: elements sharing a path keep only the last one under the path key (ACE then requires the line-suffixed form).
