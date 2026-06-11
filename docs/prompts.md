# Prompt versioning

Prompts are production assets, not hidden strings. Every prompt used by the system is
registered in a versioned prompt registry with declared variables, a changelog note, and
regression tests that pin its exact content.

## How it works

Two modules implement the system:

- `src/ai_engineering_showcase/prompt_registry.py` provides the generic machinery: the
  `PromptTemplate` dataclass and the `PromptRegistry` container.
- `src/ai_engineering_showcase/prompts.py` defines the actual production prompts and
  registers them in the module-level `PROMPT_REGISTRY`. The agent always renders prompts
  through this registry.

Each `PromptTemplate` carries:

| Field | Meaning |
|---|---|
| `name` | Stable prompt identifier, e.g. `rag_answer`. |
| `version` | Stable version identifier, e.g. `v1`. `latest` is a reserved alias that resolves to the most recently registered version. |
| `template` | The `str.format`-style template text with `{placeholder}` variables. |
| `required_variables` | Variables that must be provided on every render. |
| `optional_variables` | Variables with default values used when the caller omits them. |
| `changelog` | A short note describing what changed in this version. |

## Registered prompts

| Name | Version | Required variables | Optional variables (default) | Purpose |
|---|---|---|---|---|
| `rag_system` | `v1` | – | – | Citation-aware system instructions for grounded answers. |
| `rag_answer` | `v1` | `question` | `route` (`general_insight`), `context` (empty) | The main grounded RAG answer prompt with `citation: [n]` context blocks and the sectioned response format. |

## Validation

Templates are validated when they are constructed and when they are rendered:

- The declared variables must exactly match the placeholders found in the template.
  A placeholder without a declaration, or a declared variable that does not appear in
  the template, raises `PromptVariableError` immediately.
- Rendering without a required variable, or with an unknown variable, raises
  `PromptVariableError` with a message naming the prompt, the version, and the
  offending variables.
- Looking up an unknown prompt name or version raises `PromptNotFoundError` listing
  what is available.

## CLI

List all registered prompts with versions, variables, and changelog notes:

```bash
poetry run ai-showcase prompts list
```

Render a prompt. `--var key=value` can be repeated; optional variables fall back to
their defaults when omitted:

```bash
poetry run ai-showcase prompts render --name rag_answer --version latest \
  --var question="Why are enterprise customers unhappy with onboarding?"

poetry run ai-showcase prompts render --name rag_answer --version v1 \
  --var question="Why is onboarding slow?" \
  --var route=onboarding \
  --var "context=citation: [1]\nsource_id: fb-001\ntext: setup took weeks"
```

Missing required variables produce a clear error and a non-zero exit code.

## Prompt regression tests

`tests/test_prompt_snapshot.py` pins the main RAG answer prompt at the byte level
against golden snapshots in `tests/snapshots/`:

- `rag_answer_v1_template.txt` is the raw registered template.
- `rag_answer_v1_rendered.txt` is a fully rendered grounded prompt for fixed inputs,
  produced through `build_grounded_prompt` (so context-block construction is covered too).

Any edit to the prompt text — intentional or accidental — fails these tests.
`tests/test_prompt_registry.py` covers the registry mechanics, including that old
versions remain available after a new version is registered.

## Changing a prompt

1. Do not edit a registered template in place. Register a new `PromptTemplate` with the
   next version (e.g. `rag_answer` `v2`) and a changelog note in `prompts.py`. The
   previous version stays available via `PROMPT_REGISTRY.get("rag_answer", "v1")`.
2. Point the code path (e.g. `build_grounded_prompt`) at the new version deliberately.
3. Add new snapshot files for the new version and update the snapshot tests. The `v1`
   snapshots keep guarding the old version.
4. Run the evaluation harness (`poetry run ai-showcase evaluate`) to measure the impact
   before shipping the change.
