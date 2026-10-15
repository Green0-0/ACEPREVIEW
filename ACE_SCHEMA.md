# ACE

ACE is a worker-graph runtime (`ace.graph`) and a NEAT evolution loop (`ace.neat`) on top of the agent2 library. A worker graph is a typed DAG of LLM workers that together turn one task prompt into one proposed edit of a repository. NEAT evolves the graph structure and the workers' categorical prompt and budget settings. `GRAPH_NEAT_DESIGN.md` is the binding interface contract; this document describes the implemented behaviour.

## Relationship to agent2

**Reused:**

ACE does not modify agent2. It uses the `Session` as the pristine repository and rollout store, `Workspace` overlays as private worker sandboxes, `Agent` with an in-memory `AgentConfig` for every LLM call, the viewing, searching and editing tools from `agent2.core.globals.TOOLS`, the `default` and `directory` constructors, the `tool_call` hook, and `code_parser` (`CodeFile`, `CodeNode`, `CodeState`, `FileEdit`, the Python adapter and its Tree-sitter tree). `agent2.utils.codeblocks.extract_all_codeblocks` parses the ListViewer selection.

**Added:**

ACE adds canonical views and their rendering, the `Edit` payload with a git-faithful patch, the ACE code-block editing format, a codebase graph with personalized PageRank, view recording around agent2 tools, worker types with property tables and prompt texts, a retrying LLM client, the graph runner, genomes, an innovation database, mutation, crossover, speciation, a population, JSONL history, evaluators, and two command-line tools.

**Agent Construction:**

`Worker.make_agent` builds an `AgentConfig` with path `graph/{type}/{id}`, subagent name `{type}_{id}`, prompt wrappers `{{PROMPT}}`, `workspace_policy` ISOLATED, no subagents, hooks `["tool_call"]` only when tools are present, and constructors `["default"]` plus `"directory"` when tools are present. It passes `client=ctx.client_factory()` and a fresh `Workspace(ctx.session)`. `delegate`, `message_subagent`, `cancel_subagent`, `bash` and `tools` are forbidden and raise `ValueError`. Tool-using agents get a `ViewRecorder` and their `agent.tools` are replaced by recording wrappers.

## Payloads and Workers

**Payload Kinds:**

| Kind | Payload | Meaning |
|------|---------|---------|
| VIEW | `View` | What a worker was shown or looked at, against pristine. |
| EDIT | `Edit` | A complete proposed change against pristine. |
| REPORT | `str` | The worker's final free-text message. |

**Worker Types:**

| Type | VIEW in | EDIT in | REPORT in | Outputs | Tools | LLM steps |
|------|---------|---------|-----------|---------|-------|-----------|
| VIEWER | none | none | none | VIEW, REPORT | viewing, search | `max_steps` |
| LISTVIEWER | 1..inf | none | none | VIEW, REPORT | none | 2 |
| EDITOR | 0..inf | none | 0..inf | VIEW, EDIT, REPORT | viewing, search, editing | `max_steps` |
| VIEWEDITOR | 1..inf | none | 0..inf | VIEW, EDIT | none | 1 (ACE) |
| CRITIC | 1..inf | 1..1 | 0..inf | VIEW, EDIT | none | 1 (ACE) |
| JUDGEMERGER | 1..inf | 2..inf | 0..inf | VIEW, EDIT | none | 1 (a number) |
| PICKMERGER | 1..inf | 2..inf | 0..inf | VIEW, EDIT | none | 1 (ACE) |

`types.py` holds the table as `WORKER_SPECS` (`SlotSpec(kind, min, max)`, `max=None` unbounded) and derives `EDIT_PRODUCERS`, `VIEW_PRODUCERS`, `REPORT_PRODUCERS` and `MERGER_TYPES`.

**Inputs:**

A worker receives exactly the payloads on its incoming edges, grouped by source node id in `WorkerInputs.views`, `.edits` and `.reports`, never flattened. `WorkerInputs.source_types` maps each source id to its worker type for labelling.

**Outputs:**

A forwarded VIEW is what the worker was actually shown after expand-and-contract, merged for tool users with everything recorded from their tool calls. ACE-producing workers and JUDGEMERGER emit no REPORT. VIEWER, LISTVIEWER and EDITOR fail on an empty report. Each worker returns its agent's rollout id in `WorkerOutputs.rollout_ids`.

## Graph Invariants

**Structure:**

A `WorkerGraph` holds `nodes` (id to `WorkerNode(id, type, properties)`) and `edges` (`WorkerEdge(src, dst, kind)`). The end node is the string `END`. There is no start node; every worker receives the task prompt.

**Validation:**

`validate()` checks in order: node table consistency (key equals `node.id`, integer ids); edge legality (endpoints exist, no self-loops, no duplicate `(src, dst, kind)`, kind in the source's outputs and in the destination's input table, END accepts only EDIT); acyclicity; END has exactly one incoming edge and it is EDIT; every worker's incoming count per kind lies in its slot, and kinds without a slot have none; every worker reaches END through edges of any kind. The first violation raises `InvalidGraph` naming the node (`node 3 (critic)`) or edge (`edge 1->END [view]`).

**Order and Hash:**

`topological_order` is Kahn's algorithm with ties broken by the smallest id. `structure_hash` is Weisfeiler-Lehman colour refinement: colours start as worker types, each round hashes a node's colour with the sorted `(kind, colour)` multisets of its in- and out-neighbours, and the digest covers sorted node colours and edges. It ignores ids and properties. `minimal_graph()` is one EDITOR (id 0) feeding END.

## Pristine Base and Isolation

**Pristine Rule:**

The `Session` files at run start are the base of every view, edit, diff, rendering and ACE target. A worker never sees another worker's edited state; it sees other workers' edits only as diffs against pristine.

**Workspace Isolation:**

Every agent gets its own fresh `Workspace(session)`. Workspaces are never shared, cloned or committed. The END payload is returned as an `Edit`; the caller decides what to do with it.

**Disk Writes:**

The runtime writes only agent rollouts, through the Session's rollout directory. Callers place it with `Session(project, rollout_dir=...)`; an absolute path escapes the project folder. `RunContext.rollout_dir` records that path and is informational.

## Views

**Canonical Form:**

`View.files` maps a path to `FileView(path, levels, line_ranges)`. `levels` maps an element path in `CodeNode.path` form (`Class.method`) to `Level.COLLAPSED` (1) or `Level.FULL` (2); the key `""` is the file itself. `line_ranges` are 1-indexed inclusive ranges, sorted, with overlapping and adjacent ranges merged. `add_element` strips a trailing all-digit `llm_path` segment. `add_lines` swaps a reversed range, clamps the start to 1 and ignores a range ending before line 1. `merge` takes the maximum level and unions ranges.

**Restriction:**

`restrict_to(session)` normalises element keys through `CodeFile.code_nodes`, drops unknown files and elements and element keys on unparsed files, clamps ranges to the file length, and drops empty file views.

**Counts:**

`counts()` maps `(path, element)` to `int(level)`: FULL 2, COLLAPSED 1. Each line range adds 1 to the file key `(path, "")`. Summing across parents is the consumer's job; `CodebaseGraph.seeds_from_views` does it and also resolves ranges to elements.

**JSON:**

`{"files": {path: {"levels": {element: 1|2}, "line_ranges": [[a, b]]}}}` with sorted keys. Level names are also accepted on read.

## Rendering

**Line Sets:**

Rendering is line based. For an element, its head is the signature lines (decorators included) plus the docstring. FULL shows every line of the element. COLLAPSED shows the head plus the signature lines of each direct child. Any listed element also shows the heads of its ancestors. The file skeleton (`levels[""] == COLLAPSED`) shows every line outside top-level elements plus the head of each top-level element. `levels[""] == FULL` shows the whole file. The union of all line sets is printed.

**Header Formats:**

```
--- path ---                 whole file or skeleton (levels[""] present)
--- path::Element.path ---   one element section when levels[""] is absent
--- path (lines a-b) ---     a raw line range not already shown
```

Without `""`, listed elements are grouped under their outermost listed ancestor, one section each, ordered by start line. Line-range sections follow; lines already shown are subtracted and each remaining span becomes its own section.

**Line Prefix and Hidden-Body Marker:**

Shown lines are prefixed `N| `. A run of hidden lines becomes one `... [BODY HIDDEN: N LINES] ...` line, unprefixed and indented like the first non-blank hidden line. A hidden run of only blank lines is printed as-is.

**Other Rules:**

An element key stands for every element sharing that path (property getter and setter, `if`/`else` definitions); all of them are shown. Unparsed files render whole for `""` at any level, otherwise only their ranges. `render_view` renders files sorted by path, skips missing files and empty renderings, and joins sections with a blank line. `measure(text)` is `len(text)`: budgets are characters. The agent2 depth-based renderer is not used; the prefix and marker text match it.

## Edits

**Edit:**

`Edit.files` maps a path to the complete new bytes, or `None` for a deletion. `from_workspace` captures checkouts whose current bytes differ from their original. `to_workspace(session)` writes the bytes onto a fresh `Workspace` and ignores deletion of a path the session lacks. JSON stores bytes as base64. No patch is ever applied inside ACE.

**Git Patch:**

`per_file_diff(session)` builds one diff per changed file against pristine bytes; `git_patch` joins them with `"\n"` and `unified_diff` is its alias. Each file diff starts with `diff --git a/p b/p`, then `new file mode 100644` or `deleted file mode 100644`, then `--- a/p` or `--- /dev/null` and `+++ b/p` or `+++ /dev/null`, then hunks from `difflib.unified_diff` with 3 context lines. Content is decoded as UTF-8 with `surrogateescape` and split on `\n` only, keeping terminators, so `\r\n` and lone `\r` survive. A line without a terminator is followed by `\ No newline at end of file`. Creating or deleting an empty file has no hunks and no `---`/`+++` lines. Every file diff ends with `\n`. Binary content is diffed as text and paths are unquoted. `git apply` accepts the result.

## ACE Editing Format

**Purpose:**

Tool-less editing workers reply with markdown code blocks. The first line inside a block is a header beginning with `@@ `; the remaining lines are the body.

**Header Grammar:**

```
@@ path                      FILE     replace the whole file
@@ path::NEW                 NEW      create a file that does not exist
@@ path::DELETE              DELETE   delete the file; body ignored
@@ path::A-B                 LINES    replace lines A..B inclusive (1 <= A <= B)
@@ path::A                   LINES    same as A-A
@@ path::Elem.path           ELEMENT  replace the element, decorators and signature included
@@ path::Elem.path::AFTER    AFTER    insert the body as a new sibling after the element
@@ path::Elem.path::BEFORE   BEFORE   insert the body as a new sibling before the element
```

Parts are split on `::` and stripped; an empty part is malformed. Keywords are upper case. A leading `./` is dropped and backslashes in the path become `/`. Element paths match `[A-Za-z_][\w.]*` and may be `CodeNode.path` or `llm_path` (`Item.slug.14`); a stale line suffix falls back to the path form.

**Formats:**

| `ace_format` | Permitted targets |
|--------------|-------------------|
| `element` | ELEMENT, NEW, DELETE, AFTER, BEFORE |
| `element_file` | + FILE |
| `element_file_lines` | + FILE, LINES |
| `file_lines` | FILE, NEW, DELETE, LINES |

`ace_instructions(fmt)` produces the prompt text for the permitted headers.

**Fences:**

A fence is three or more backticks, optionally indented and tagged, closed by a line of at least as many backticks and nothing else. Content lines lose up to the opening fence's indentation. Blocks whose first line does not start with `@@ ` are ignored, and so are quoted unified-diff hunks (`@@ -12,6 +12,7 @@`). A message with no ACE block, an unclosed fence that starts with a header, a malformed header, or a header not permitted by the format raises `AceError`.

**Application:**

All targets resolve against pristine `session.files`. ELEMENT replaces from the start of the element's first line to its end byte with the body re-indented by `adapter.attempt_fix_formatting`; a body already at the target indentation that contains less-indented lines is kept as given. AFTER inserts at the end of the element's last line and BEFORE at the start of its first line, indented like the element, separated by one blank line when nested and two at top level. LINES replaces whole lines and appends a newline if the replaced span ended with one and the body does not; an empty body deletes the lines. FILE and NEW write the body verbatim. Text inserted into a mostly `\r\n` file uses `\r\n`. Per file, edits are applied bottom-up by (start, end) descending on a fresh workspace and the result is `Edit.from_workspace`.

**Errors and Conflicts:**

`AceError` subclasses `GraphFailure`. It is raised for an unknown path or element, an element header on an unparsed file, a line span outside the file, NEW on an existing path, and a path shared by several elements addressed without its line suffix (the message lists the `llm_path` choices). Byte ranges in one file must not overlap as half-open ranges. A zero-width insertion conflicts with a range strictly containing its point, with another insertion at the same point, and with a replacement that strictly contains its anchor element. A FILE, NEW or DELETE block must be the only block for its path.

## Codebase Graph

**Nodes:**

`NodeKey = (path, element)`; element `""` is the file node. Every file in the pristine session contributes its file node; a `CodeFile` contributes one node per unique `CodeNode.path`. The graph is built once per run, lazily, by `RunContext.codegraph`.

**Edges:**

| Edge | Direction | Weight |
|------|-----------|--------|
| containment | file -> top-level element, element -> direct child | 1.0 |
| containment back | child -> parent | 0.5 |
| sibling | consecutive elements with the same parent, both ways | 0.3 |
| import | Python file -> imported project file | 1.0 |
| name reference | element -> element named in its body | 0.7 / n |

Parallel edges are summed. Imports are read from the Tree-sitter tree, including nested imports. `a.b.c` resolves to `a/b/c.py` or `a/b/c/__init__.py`, else to a unique file whose module path ends with `a.b.c`. `from M import n` tries `M.n` before `M`; relative imports resolve against the importing package. Name references take `identifier` tokens attributed to the innermost enclosing element body, excluding definition names, import statements, names shorter than 3 characters, keywords, soft keywords, builtins, `self` and `cls`. A name matching `n` elements adds `0.7 / n` to each; repeats add nothing.

**Personalized PageRank:**

`rank(seeds, damping=0.85, iterations=30)` runs power iteration over out-weights normalised per node. Teleport and dangling mass go to the seed distribution. Non-positive and unknown seeds are dropped; with none left, seeds are uniform over file nodes. Every node is returned, sorted by descending score then key.

**Budgeted Selection:**

`select(seeds, budget, session)` walks positive seeds by descending weight, then the remaining nodes in rank order. Elements are added at FULL, file nodes at COLLAPSED, and an element's file is added at COLLAPSED with it. Costs are `size_of` estimates in characters including `N| ` prefixes; the file skeleton is paid once per file and lines already paid for are not charged again. A candidate that does not fit is skipped; the walk stops after 50 consecutive skips. The result is a `View`, never text.

**Seeds and Listings:**

`seeds_from_views(views, full_weight=2)` sums `counts()` over the parent views, maps a FULL sighting to `full_weight`, and adds 1 to every element a line range intersects (`resolve_lines`). `candidate_list(view_or_keys, session, budget)` prints `path::element  <signature line>` or `path` per node, in rank order for a View, whole lines only, within `budget` characters.

## View Recording

**Recorder:**

`ViewRecorder` keeps a log of sightings against the file state at call time. `record_file(path, full)` records the file at FULL or COLLAPSED. `record_element` records FULL. `record_at(path, line)` records the innermost element containing the line at FULL, else lines `line..line+49` (to the end of file for `full=True`). `record_lines` stores the bytes of the state the lines refer to.

**Tool Wrappers:**

`recording_tools(names_or_mapping, recorder)` wraps the named global tools or a `name -> Tool` mapping such as `agent.tools`, preserving per-agent schemas. A `RecordingTool` exposes `func`, `name`, `description`, `openai_schema` and `examples`. Viewing tools (`view`, `view_path`, `view_files`, `view_file`, `views_at`, `view_at`, `view_element`, `view_multi_lines`, `view_lines`) and `open_*` tools record when their result does not start with `Error`; multi-item tools re-check each item with the single-item viewing function. `open_*` targets count as FULL. A recording failure is logged and never changes the tool result.

**Pristine Remap:**

`finalize(workspace)` replays the log onto pristine. Sightings of files absent from pristine are dropped. File sightings keep their level. Element sightings are looked up by path in the pristine file and dropped when missing. Line ranges map from the seen bytes to pristine bytes: identical bytes keep the range; otherwise `SequenceMatcher(None, pristine, seen, autojunk=False)` opcodes are used, lines in `equal` blocks map by offset and all others are dropped. The result is passed through `restrict_to`.

## Properties and Prompts

**Categorical Genes:**

A node stores `property name -> option index`. Index 0 is the default. Mutation picks a property with more than one option and a different index; crossover picks one parent's index per name. `validate_properties` rejects unknown, missing or out-of-range entries; `resolve` maps indices to values.

**Property Table:**

| Property | Types | Options (index 0 first) |
|----------|-------|-------------------------|
| `role` | all | 4 texts per type |
| `strategy` | all | 4 texts per type |
| `report_focus` | VIEWER, LISTVIEWER, EDITOR | 4 texts per type |
| `instruction` | all | 4 texts per type |
| `context_budget` | all but VIEWER | 12000, 6000, 24000, 48000 |
| `max_steps` | VIEWER | 20, 10, 40 |
| `max_steps` | EDITOR | 30, 15, 60 |
| `tool_set` | VIEWER | viewer_standard, viewer_minimal, viewer_lines, viewer_full |
| `tool_set` | EDITOR | editor_standard, editor_minimal, editor_interactive, editor_full |
| `ace_format` | VIEWEDITOR, CRITIC, PICKMERGER | element, element_file, element_file_lines, file_lines |
| `list_budget` | LISTVIEWER | 8000, 4000, 16000 |
| `view_full_weight` | LISTVIEWER, VIEWEDITOR, CRITIC, JUDGEMERGER, PICKMERGER | 2, 1, 3 |

EDITOR has no `view_full_weight`; it seeds input views with weight 2.

**Prompt Slots:**

`prompts.py` holds `ROLES`, `STRATEGIES`, `REPORT_FOCI` and `INSTRUCTIONS`, each a per-type tuple of four meaningfully different texts. Strategies are viewing approaches for viewers, selection approaches for the ListViewer, editing disciplines for editors, review rubrics for the Critic, and judging or merging rubrics for mergers. Tool sets never contain the forbidden tools.

**Assembly:**

`build_system_prompt` joins role, strategy, report focus when present, and the type's output-format block, separated by blank lines; tool-using types end with `{{FILES*}}` for the `directory` constructor. ACE blocks carry the literal `{ace_instructions}`, filled with `str.replace`; prompt text is never passed to `str.format`. The user message is the `instruction` text, then `# Task` with the prompt, then non-empty sections: `# Code (original repository)`, `# Candidate edit under review ...`, `# Candidate edits ...`, `# Reports from other workers`. Reports are labelled `REPORT from worker {id} ({type})` and candidates `CANDIDATE {n} (worker {id})` followed by a fenced diff or `(no changes)`.

## Runtime

**RunContext:**

`RunContext(session, prompt, settings, client_factory, codegraph=None, run_id=uuid, rollout_dir=None)`. `ModelSettings(model, api_base_url=None, api_key_env_var="ANTHROPIC_API_KEY")` is defined in `llm.py` and re-exported by `runner.py`.

**Execution:**

`GraphRunner(graph)` validates the graph. `run(ctx)` repeatedly takes every unrun worker whose sources have finished, starts them concurrently, and waits until all finish or one fails. Properties are resolved right before a worker runs. After a worker finishes, every outgoing edge's payload must be non-`None`. The result is `GraphResult(edit, diff, outputs, order)` where `edit` is the END source's edit, `diff` its git patch, and `order` the start order.

**Worker Behaviour:**

VIEWER runs its tool agent and outputs the finalized recorded view and report. LISTVIEWER lists candidates from `select(seeds, list_budget)` (falling back to the whole ranked graph when empty), parses the selection reply into a View (code-block lines, else all lines; bullets, numbering, backticks and trailing signatures stripped; unresolvable entries dropped), fails if nothing resolves, expands the selection to `context_budget`, merges it with the selection, renders it as a followup and takes the second reply as its report. EDITOR adds rendered input views and reports to its prompt and outputs the recorded view merged with the rendered one, `Edit.from_workspace` and its report. VIEWEDITOR, CRITIC and PICKMERGER render inputs, run one step and apply the ACE reply; they fail on an empty rendering. CRITIC shows its single input edit with the author's report beneath it and replaces that edit entirely. JUDGEMERGER takes the first integer in its reply as the candidate number and forwards that `Edit` object unchanged.

**Failure Policy:**

Any worker failure aborts the task run with `GraphFailure`; the task scores zero and the genome stays in the population until selection removes it. Format violations, unparseable selections, ACE errors, wrong input counts, empty reports and missing payloads all raise. A non-`GraphFailure` exception is wrapped as `node N (type) failed: ...`. Every worker failure carries `node_id`. Sibling tasks are cancelled; if several fail, the lowest node id wins.

**Retry Layer:**

`make_client_factory(settings, max_retries=5, backoff=1.0)` returns a factory of `RetryingClient` proxies around `AsyncOpenAI(max_retries=0)`. `chat.completions.create` retries connection errors, timeouts, 429 and any status of 500 or more, waiting `min(backoff * 2**attempt, 30)` seconds, for at most `max_retries + 1` calls. Other 4xx errors raise at once. Transient errors are never handled by graph redundancy.

## Genome and Innovations

**Genome:**

A `Genome` holds `nodes` (innovation to `NodeGene(innovation, type, properties)`), `connections` (innovation to `ConnectionGene(innovation, src, dst, kind, enabled)`), `id`, `generation`, `parents`, `operators`, `fitness`, `adjusted_fitness` and `species_id`. Genome and `Species` use identity equality and hash by identity. `to_graph` keeps every node gene and the enabled connections. `validate` checks gene-level consistency (keys, types, endpoints of enabled and disabled genes, unique triples, connection innovations disjoint from node innovations) and then the graph. `assert_types_match(other)` checks that shared innovations denote the same thing in both genomes. `num_workers` is the number of node genes.

**Innovation Database:**

`InnovationDB` hands out node and connection innovations from one counter starting at 0, so ids never collide. Keys are never reset. Connections are keyed `(src, dst, kind)`. It saves to JSON with tagged key elements so enum types round-trip.

**Node Key Families:**

| Event | Key |
|-------|-----|
| initialisation | `("init", index, type)`; index is the genome's node count at creation |
| LISTVIEWER split of a VIEW edge | `("split", conn_innovation, LISTVIEWER)` |
| CRITIC split of an EDIT edge | `("critic", conn_innovation, CRITIC)` |
| VIEWER source feeding `d` | `("source", d, Kind.VIEW, VIEWER)` |
| merger in front of END | `("merge", end_source, candidate, JUDGEMERGER or PICKMERGER)` |
| VIEWEDITOR mirroring `d`'s VIEW sources | `("mirror", d, VIEWEDITOR)` |
| EDITOR fed by VIEW producer `v` | `("branch", v, EDITOR)` |

The fallback VIEWER created to give a new CRITIC or merger a VIEW input uses the source key with `d` set to that critic or merger. During initialisation every created node uses the init key instead. A proposal whose key yields an innovation already in the genome is rejected.

## Mutation

**Operators:**

Each operator mutates in place, returns `True` on success, and leaves the genome unchanged on failure. A proposal runs on a copy, and up to `max_attempts` proposals are tried until one validates. `mutate_properties` mutates each node with `p_property_per_gene` and forces one node if none was picked. `add_connection` adds or re-enables a legal edge that creates no cycle and never targets END. `remove_connection` deletes an enabled gene. `toggle_enable` flips any gene. `remove_worker` never removes the END source; a merger left with one EDIT input is collapsed, its remaining candidate inheriting every outgoing edge of a kind it outputs. New workers beyond `max_workers` are rejected.

**Typed Insertion Table:**

| Row | Site | Inserted worker and wiring |
|-----|------|----------------------------|
| listviewer | enabled VIEW edge `s->d` | LISTVIEWER `n`: `s->n` VIEW, `n->d` VIEW; original disabled |
| critic | enabled EDIT edge `s->d` (d may be END) | CRITIC `n`: `s->n` EDIT, `n->d` EDIT; VIEW from `s`, else a VIEW-producing ancestor, else a new VIEWER; original disabled |
| viewer | node `d` with a VIEW slot | VIEWER `n`: `n->d` VIEW, plus REPORT with probability 0.5 if `d` accepts it |
| vieweditor | node `d` with VIEW inputs | VIEWEDITOR `n` fed by `d`'s VIEW sources; `n->` a random non-ancestor merger, else merger insertion with `n` as candidate |
| editor | VIEW producer `v` not feeding an EDITOR | EDITOR `n`: `v->n` VIEW (+ REPORT if `v` outputs it); merger insertion with `n` as candidate |

`add_worker` picks a row uniformly per attempt, then a random site.

**Merger Insertion:**

`insert_merger` joins the END source `e` and a candidate `c` whose edit does not reach END: an existing one, else a new EDITOR or VIEWEDITOR from the table. It creates a JUDGEMERGER or PICKMERGER `m` with `e->m` EDIT, `c->m` EDIT and `m->END` EDIT, disables `e->END`, adds VIEW from the VIEW-producing members of `{e, c}` (else an ancestor, else a new VIEWER), and REPORT from each of `e`, `c` that outputs it with probability 0.5.

**mutate:**

`mutate` copies the genome with a new id, resets fitness and species, applies `mutate_properties` with `p_property` and each structural operator with its probability, and appends applied names to `operators`. An invalid result falls back to a copy with `"noop"`. `random_valid_genome` starts from the minimal genome with random editor properties and applies `add_worker` (75%) or `insert_merger` (25%) with init keys until a worker count drawn from `[min, max]` is reached.

## Crossover and Repair

**Alignment:**

`crossover(a, b, rng, *, innovations, max_workers=None)` aligns node and connection genes by innovation. Fitness compares adjusted fitness when both are set, else raw. Matched nodes take each property from a random parent. A matched connection disabled in either parent is disabled with probability 0.75, else enabled. Disjoint and excess genes come from the fitter parent, or from both when equal. The child's generation is one more than the larger parent generation and its operators are `["crossover"]`. No innovations are allocated.

**Repair:**

An invalid child is repaired in steps, stopping at the first valid state: drop connections with missing endpoints; fix END by enabling one disabled EDIT input or keeping one of several enabled, preferring the fitter parent's END source; remove nodes that cannot reach END, to a fixed point; remove nodes whose required slots are unmet, re-applying the reachability step, to a fixed point. A still-invalid child, or one with more than `max_workers` workers, becomes a copy of the fitter parent with `operators == ["crossover_fallback"]` and fitness reset.

## Speciation

**Distance:**

`delta = c_excess * E + c_disjoint * D + c_properties * H`. E and D count non-matching genes over the union of node and connection innovations, enabled or not; a gene is excess when above the other genome's largest innovation. N is always 1: genomes are small and the classic switch to gene counts at 20 genes collapsed species mid-run. H is the mean, over matched nodes, of the fraction of differing property indices.

**Assignment:**

At the start of `speciate`, each species picks a random previous member as representative and clears its members. A genome joins the first species by ascending id within the threshold, or founds a new one. Species ids are never reused and empty species are dropped. With speciation disabled everyone is in species 0.

**Threshold:**

`set_threshold_for(representatives)` sets the threshold just below the smallest pairwise representative distance. `adjust_threshold` is relative: fewer species than `target_species` multiplies it by `1 - threshold_step`, more by `1 + threshold_step`. The floor is 0.05.

**Sharing and Allocation:**

`share_fitness` sets `adjusted = (adjusted + shift) / species size`, where `shift = -min adjusted` over the population when negative, else 0. Raw fitness is never changed. `allocate_offspring` divides the total proportionally to summed shifted adjusted fitness (member counts when all are zero) with largest-remainder rounding.

**Stagnation:**

`prune_stagnant` updates each species' best raw fitness and `last_improved` on strict improvement, then removes species with `generation - last_improved > stagnation_generations`, except the species holding the best raw genome. Nothing is pruned before any genome has a raw fitness.

## Population Lifecycle

**Initialisation:**

`initialize` draws `population_size` genomes with `random_valid_genome` in `initial_min_workers..initial_max_workers`, re-rolling up to 10 times on exact duplicates, with `operators == ["init"]`. The innovation DB comes from the history's `innovations.json` when present.

**Evaluation:**

`evaluate` scores genomes without fitness concurrently, sets `adjusted = raw - node_cost * num_workers`, and logs each genome once after scoring. Adjusted fitness is recomputed from raw at the start and end of each step, so logged values are never shared values.

**Bootstrap:**

After gen-0 evaluation, `bootstrap_species` chooses K representatives greedily by adjusted fitness (ties: fewer workers), skipping genomes within 0.05 of a chosen one. The rest of the population is regenerated as mutants of the representatives in round robin and evaluated. The threshold is set from the representatives, each founds one species, and every other genome joins its nearest representative's species.

**Step:**

`step` increments the generation, speciates, adjusts the threshold, shares fitness, prunes stagnant species and allocates offspring. Per species it keeps `min(elitism_per_species, quota)` elites unchanged and unevaluated, takes the top `max(1, ceil(survival_fraction * size))` as survivors, and fills the quota with crossover of two survivors (probability `crossover_rate`) followed by mutation, or mutation of one survivor. Every child carries at least one mutation operator. A shortfall is filled with mutants of the best genome. Children are evaluated and stats logged. `run` initialises, evaluates and bootstraps an empty population, then steps until `cfg.generations`. `best` is the highest raw fitness, ties to fewer workers.

**State:**

`save_state` writes one JSON file with the config, generation, genomes, speciator threshold and species, next species id, innovation DB, RNG state and last elites. `load_state` restores it in place.

## History

**Files:**

`History(directory)` appends JSONL under `history_dir` (default `.ace/neat`, created on first write):

```
genomes.jsonl      {"id", "generation", "parents", "operators", "species_id", "fitness",
                    "adjusted_fitness", "structure_hash", "num_workers", "genome": <to_json()>}
species.jsonl      {"generation", "threshold", "species": [{"id", "size", "member_ids",
                    "representative_id", "best_fitness", "created", "last_improved"}]}
generations.jsonl  {"generation", "population", "best_fitness", "mean_fitness", "median_fitness",
                    "best_adjusted", "mean_adjusted", "median_adjusted", "num_species",
                    "species_sizes", "threshold", "mean_workers", "best_id", "best_workers",
                    "num_elites", "elites", "evaluated"}
innovations.json   {"version": 1, "next_id", "nodes": [{"key", "id"}],
                    "connections": [{"src", "dst", "kind", "id"}]}
```

**Rules:**

A genome id is logged once, including ids found in an existing file. `species.jsonl` records the bootstrap species as generation 0 and, for each step, the species used to breed it. Malformed lines are skipped with a warning and the next append starts on a fresh line. `lineage(id)` returns the genome's record first and its logged ancestors breadth-first, oldest last.

## Evaluation

**MockEvaluator:**

A structural, network-free score:

```
score = 1.00 * |distinct worker types|
      + 0.50 * [END source is a JUDGEMERGER or PICKMERGER]
      + 0.25 * |CRITIC workers|
      - 0.10 * max(0, workers - 6)
      + noise * (2u - 1)
```

`u` comes from `random.Random(int(structure_hash, 16))`, so noise is a fixed function of structure. A `scores` mapping keyed by structure hash overrides the formula. Properties never affect the score.

**TaskEvaluator:**

`TaskEvaluator(tasks, settings, client_factory=None, concurrency=4, rollout_dir=None)` runs the genome's graph on every `Task(project_dir, prompt, scorer)` and returns the mean score. Each task gets a fresh `Session` and `RunContext`. One semaphore bounds concurrent task graphs across all concurrent evaluations. A `GraphFailure` scores 0.0 and is logged; scorer and Session errors propagate. `last_results` and `results_by_genome` hold per-task records with `task`, `project_dir`, `score`, `failure`, `node_id`, `diff_chars` and `rollout_dir`.

**Rollout Isolation:**

Task `i` of genome `g` writes rollouts to the absolute directory `<root>/<g.id>/<i>/`, used both as the Session's rollout dir and as `RunContext.rollout_dir`. `root` is the constructor's `rollout_dir`, or a private temporary directory created on first use and kept. Each Session therefore starts with no rollouts to load, and nothing is written into a project folder. A root inside a task's project on a non-hidden path raises `ValueError`, because the session would read the rollouts as project files.

**Scorers:**

A scorer is called as `scorer(edit, session)` after a successful run, with the END `Edit` and the task's pristine `Session`; it may be sync or async and returns a float. `load_tasks(path)` reads a JSON list of `{"project_dir", "prompt", "scorer"}`; relative project dirs resolve against the file's directory. `scorer` is `"nonempty_diff"` (the default; 1.0 if the edit changes anything) or `"package.module:function"`. A SWE-bench scorer plugs in as such a function: it materialises the edit with `edit.to_workspace(session)` or writes `edit.git_patch(session)` (encoded with `surrogateescape`) into a checkout and runs the instance's test harness.

## Command-Line Tools

**run_graph.py:**

`uv run python scripts/run_graph.py <folder> <prompt>` runs one graph on a project and prints each worker's rollouts, view size, edit paths and report excerpt, then the final git patch. Flags: `--graph FILE` (WorkerGraph JSON; default `minimal_graph()`), `--model` (default `claude-haiku-5-5`), `--api-base` (default `https://api.anthropic.com/v1/`), `--api-key-env` (default `ANTHROPIC_API_KEY`), `--rollout-dir DIR` (default `<folder>/.agent2/task`), `--print-views`. The key is loaded from `.env` at the ACE root, then `agent2/.env`, when not already set. It exits 1 on `GraphFailure` or an invalid graph.

**run_neat.py:**

`uv run python scripts/run_neat.py` evolves graphs. Flags: `--config TOML` (NeatConfig; top-level fields with `[speciation]` and `[mutation]` tables, or one `[neat]` table), `--mock`, `--mock-noise` (default 0.1), `--tasks FILE`, `--model`, `--api-base`, `--api-key-env`, `--concurrency` (default 4), `--history-dir`, `--generations`, `--population` (also caps `target_species`), `--seed`, `--resume STATE`, `-v`. One of `--mock` or `--tasks` is required. Task rollouts go to `<history-dir>/rollouts/<genome id>/<task index>/`. After every generation it prints a summary line and saves `<history-dir>/population_state.json`; at the end it writes `<history-dir>/best_genome.json` (genome JSON plus `"graph"`).

## Known Limitations

- The codebase graph is Python-only: imports and name references come from the Python Tree-sitter tree, and other files contribute only file nodes.
- agent2's `Workspace.diff` is unfaithful (it uses `str.splitlines`, drops terminators and the no-newline marker), so ACE builds its own git patch.
- agent2's `Session.refresh_files` runs on every tool call and re-reads the project, so the pristine base can change if something writes the project folder mid-run.
- Elements that share a path (property getter and setter, `if`/`else` definitions) require the line-suffixed `llm_path` form in ACE headers.
- agent2's `replace_lines` corrupts the file when given a reversed range; it does not check `line_start <= line_end`.
- agent2's `attempt_fix_formatting` splits with `str.splitlines`, so it also splits on form-feed characters inside a body.
- agent2's `DirectoryConstructor` substitutes `{{FILES*}}` anywhere in the chat, including the task prompt.
- agent2's `make_empty_files` truncates existing files.
- Live-model behaviour has only been exercised with fakes; no run against a real API is covered by the test suite.
