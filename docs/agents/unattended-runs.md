# Unattended runs

`/implement-spec` usually runs overnight with no one watching. These rules apply to the orchestrator and to every subagent it starts, and they override any skill step that says to ask or confirm with the user.

## Decide, log, keep going

When a choice would normally go to the user (a missing tool, an ambiguous ticket, a trade-off between approaches, a skill asking to confirm), take the recommended option and keep going.

Record each decision in the **decision log**: `DECISIONS.md` in the run's shared notes directory, outside the repo. The orchestrator creates it at the start of the run and gives its path to every subagent. Each entry follows this format:

```
## <ticket number, or "orchestrator">: <the question>
- Options: ...
- Chose: <one sentence>
- Why: <one sentence>
- Detail: ... (optional: lists, figures, file names)
```

A **record** (a real run's command, duration and output size) is not a decision: head it `## <ticket>: record: <what ran>` and put its facts under `- Detail:`. The final report lists records apart from decisions.

Subagents append their own entries and list them in their report.

## The one exception

Ask the user only when every option is hard to reverse or reaches outside the repo, for example pushing to `main`, deleting data, or spending money. Before asking, finish the rest of the frontier, then ask once, at the end.

## Final report

The run ends with a report covering:
- the integration branch;
- the outcome of each ticket;
- every entry in the decision log, with its question, choice and reason, so the user can review or reverse it.
