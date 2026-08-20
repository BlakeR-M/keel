# Client overlays

An overlay is the configuration Keel needs to run for one client: which deploy profile, where the
data lives, which roles exist and which ACL tags each role carries, who the pilot users are, which
tools the agent may call, and how the client's documents map onto tags. It is config only. The Keel
code stays identical across clients; the overlay is what changes.

Two overlays ship today, one per Canberra pilot:

| Directory | Client | Roles |
| --- | --- | --- |
| `clients/example-agency/` | Example Nannying Agency (nannying agency) | owner, coordinator, carer |
| `clients/example-gym/` | Example Gym (gym) | owner, front-desk, trainer |

Both follow the same layout, so a third client is a copy of either directory with the names changed.

## The rule: client data never enters this repository

Documents, people's names, phone numbers, email addresses, rosters, waivers, agreements, exports and
databases stay on the client's machine. This repository holds placeholders and templates. The
manifest in each overlay is a template with example paths; the users list carries role names only.
Before committing anything under `clients/`, read it once more as if a stranger were reading it,
because the repository is public.

## Files in an overlay

| File | What it is | Who reads it |
| --- | --- | --- |
| `keel.yaml` | The overlay itself: client identity, `settings` (each key maps to the `KEEL_` environment variable of the same name), `roles` (role to ACL tags), `users` (pilot users with a role), `files` (the sibling files below). | People, and `tests/test_clients.py`. The runbook turns `settings` into environment variables at install. |
| `policy.yaml` | The tool policy: `allowed_tools`, `write_tools_require_approval`, `tool_arg_rules`, `max_tool_calls_per_request`. Loadable by `keel.agent.policy.Policy.from_yaml`. | Keel, once copied to `<data_dir>/policy.yaml` (`keel/providers/factory.py` loads the policy from there). |
| `corpus.manifest.yaml` | Ingest manifest template: one entry per document class with its ACL tags and an example path. | `keel ingest --manifest`, after the paths are filled in on the client's machine. |
| `runbook.md` | Install, ingest, users, approvals, admin page, backup, and what to send Blake before the pilot. | Whoever installs and runs the appliance. |
| `needs-from-client.md` | Checklist of what the client provides before install day. | Blake and the client. |

## ACL model, in one paragraph

Every chunk carries a list of tags. Every request carries a user with a list of tags. A chunk is
readable when the user holds at least one of the chunk's tags, and the check runs inside retrieval,
before anything reaches the model (`keel/retrieval/hybrid.py`). Roles are how an overlay names a
bundle of tags: the owner role holds every tag, and narrower roles hold fewer. Tag a document with the
narrowest role that should read it; wider roles carry the narrower tags too. `public` is the tag for
material anyone using the appliance may read.

## Creating a third overlay

1. Copy `clients/example-gym/` (or `clients/example-agency/`) to `clients/<client-id>/`.
2. In `keel.yaml`: set `client.id`, `client.name`, the `settings.data_dir` for their machine, the
   `roles` map and the pilot `users`. Keep the owner role holding every tag.
3. In `policy.yaml`: keep `write_tools_require_approval: true`. Trim `allowed_tools` to what the pilot
   needs. Set the `sql_readonly` table allowlist only when a reporting database is attached, and keep
   `http_get` off the allowlist for an air-gapped pilot.
4. In `corpus.manifest.yaml`: one entry per document class with example paths and tags. Real paths
   are filled in on the client's machine, never here.
5. Rewrite `runbook.md` and `needs-from-client.md` for the client's roles and document classes.
6. Add the new directory to the parametrised list at the top of `tests/test_clients.py` and run
   `python -m pytest tests/test_clients.py -q`. The tests check that the policy loads, the overlay
   references files that exist, every manifest entry carries at least one tag, and the runbook covers
   approvals, ACL and backup.

## Applying an overlay

The short version, spelled out per client in each runbook:

```powershell
$env:KEEL_PROFILE  = 'local'
$env:KEEL_DATA_DIR = 'C:\Keel\<client-id>\data'      # from keel.yaml settings.data_dir
$env:KEEL_AIRGAP   = '1'                              # from keel.yaml settings.airgap
New-Item -ItemType Directory -Force $env:KEEL_DATA_DIR | Out-Null
Copy-Item clients\<client-id>\policy.yaml "$env:KEEL_DATA_DIR\policy.yaml"
.\deploy\onprem\run.ps1 -SkipWeb
.venv\Scripts\python.exe -m keel.cli ingest --manifest C:\Keel\<client-id>\corpus.manifest.yaml
.\deploy\onprem\run.ps1
```

`deploy/onprem/run.ps1` reads `KEEL_DATA_DIR` and passes the environment through to the app, so the
appliance runs against the client's data directory and policy without any change to the repository.
