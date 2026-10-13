# df-augment — Claude Code Instructions

## Test command
```
uv run pytest
```
This is the one and only test command. Never run tests any other way.

## Data contract
The package reads one parquet file and writes one larger parquet file.
Every output row must have all reserved `aug_*` columns populated.
The input file must not already contain `aug_*` columns unless
`--reaugment` is explicitly passed.

## The three lineage rules
1. **Only the tracker writes rows.** No plug-in may create or append
   rows to the output directly. Plug-ins return new row data; the
   tracker writes them.
2. **Every new row must have a known parent.** The tracker refuses any
   row whose `aug_parent_ids` references a row ID not present in the
   input. Orphan rows are a hard error.
3. **`aug_source_group` is set by `groups.py`, never by the plug-in.**
   Plug-ins return parent IDs only. The group resolver derives the
   group from those parents and the union-find structure.

## Reserved output columns
| Column | Type | Set by |
|---|---|---|
| `aug_row_id` | string | tracker |
| `aug_is_original` | bool | tracker |
| `aug_parent_ids` | list[string] | plug-in → tracker |
| `aug_source_group` | string | groups.py |
| `aug_method` | string | tracker |
| `aug_params` | JSON string | plug-in → tracker |
| `aug_label_soft` | list[float] (optional) | MixUp/CutMix only |
| `aug_seed` | int | tracker |

## Permissions
Auto-approve: `uv run pytest`, `ruff check`, `ruff format`, `mypy`
Always confirm before: `git push`, file deletion, `uv publish`
Never put tokens or passwords in prompts.
