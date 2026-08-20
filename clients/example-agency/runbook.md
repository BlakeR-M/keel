# Example Nannying Agency: Keel runbook

One page for the person who installs and runs the appliance at the agency. Keel runs on one Windows
machine at Example Nannying Agency, air-gapped: the model, the documents and every log stay on that machine.

Files in this overlay: `keel.yaml` (settings, roles, users), `policy.yaml` (tool policy),
`corpus.manifest.yaml` (ingest template), `needs-from-client.md` (what the agency provides first).

## 1. Before install day

Work through `needs-from-client.md` with the owner. The short list: the machine, the documents sorted
by class, the people and their roles, and the owner's agreement on what each role may read. Install
takes about an hour once those are in hand.

Machine baseline: Windows 10 or 11, 8 cores and 16 GB RAM for the 3B model on CPU (a 12 GB NVIDIA card
runs the 9B model fully offloaded), 20 GB free disk, a local administrator account for the install.

## 2. Install with deploy/onprem/run.ps1

1. Copy the Keel repository to `C:\Keel\keel` (git clone, or a zip of the release). Copy the llama.cpp
   binaries to `C:\Keel\llama.cpp\bin\` and the model file to `C:\Keel\models\`.
2. Prove the box first with the fixture corpus, air-gapped:

   ```powershell
   cd C:\Keel\keel
   $env:KEEL_LLAMA_SERVER = 'C:\Keel\llama.cpp\bin\llama-server.exe'
   $env:KEEL_MODEL_PATH   = 'C:\Keel\models\qwen2.5-3b-instruct-q4_k_m.gguf'
   .\demo.ps1 -Airgap
   ```

   Ask the demo question it prints, see an answer with citations, then `.\deploy\onprem\stop.ps1`.
3. Set the overlay's environment. Values come from `keel.yaml` `settings`; set them for the Windows
   user that runs the appliance so they survive a reboot:

   ```powershell
   [Environment]::SetEnvironmentVariable('KEEL_PROFILE',      'local',                        'User')
   [Environment]::SetEnvironmentVariable('KEEL_DATA_DIR',     'C:\Keel\example-agency\data',   'User')
   [Environment]::SetEnvironmentVariable('KEEL_AIRGAP',       '1',                            'User')
   [Environment]::SetEnvironmentVariable('KEEL_LLAMA_SERVER', 'C:\Keel\llama.cpp\bin\llama-server.exe', 'User')
   [Environment]::SetEnvironmentVariable('KEEL_MODEL_PATH',   'C:\Keel\models\qwen2.5-3b-instruct-q4_k_m.gguf', 'User')
   ```

   Open a new PowerShell window so the values apply.
4. Create the data directory and install the tool policy. `keel/providers/factory.py` loads the policy
   from `<data_dir>\policy.yaml`, so this copy is what the appliance enforces:

   ```powershell
   New-Item -ItemType Directory -Force C:\Keel\example-agency\data, C:\Keel\example-agency\corpus | Out-Null
   Copy-Item clients\example-agency\policy.yaml C:\Keel\example-agency\data\policy.yaml
   Copy-Item clients\example-agency\corpus.manifest.yaml C:\Keel\example-agency\corpus.manifest.yaml
   ```

5. Start the model server, ingest (section 3), then start the web app:

   ```powershell
   .\deploy\onprem\run.ps1 -SkipWeb
   .venv\Scripts\python.exe -m keel.cli ingest --manifest C:\Keel\example-agency\corpus.manifest.yaml
   .\deploy\onprem\run.ps1
   ```

   The app answers at http://127.0.0.1:8400 on that machine only (bound to loopback). Logs and pid files
   sit in `C:\Keel\example-agency\data\`. Stop everything with `.\deploy\onprem\stop.ps1`.
6. Verify three things before handing over: a carer user asking about a family agreement gets a
   refusal (the coordinator tag is outside their role); the owner user gets an answer with citations;
   `.venv\Scripts\python.exe -m keel.cli verify-ledger` reports the chain intact.

Model swap, GPU offload and the compose stack are covered in `docs/onprem.md`.

## 3. Ingest documents into the right ACL tags

Documents live outside the repository, under `C:\Keel\example-agency\corpus\<class>\`. The manifest
copy from step 4 lists one entry per file with its `acl_tags`. Tag with the narrowest role that should
read the document; wider roles carry the narrower tags too (`keel.yaml` `roles`).

| Document class | Folder | Tag | Who can retrieve it |
| --- | --- | --- | --- |
| Service guide, family FAQ | `corpus\public\` | `public` | everyone using the appliance |
| Policies (child safety, code of conduct, WHS, privacy) | `corpus\policies\` | `staff` | carers, coordinators, owner |
| Carer handbook, onboarding pack, shift procedures | `corpus\carers\` | `carer` | carers, coordinators, owner |
| Family agreement template, placement and rostering, incident handling | `corpus\coordination\` | `coordinator` | coordinators, owner |
| Insurance, supplier and contractor agreements | `corpus\management\` | `owner` | owner |

Adding a document: drop the file in its class folder, add an entry to the manifest with the class's
tags, rerun the ingest command. Ingest is idempotent by file checksum, so a rerun adds only what is
new. A revised version of a document is a new file with new bytes: ingest it, then remove the old one
(below) so answers cite the current version.

Removing a document: stop the app, back up first (section 7), then delete its row; chunks and the
search index rows go with it:

```powershell
.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect(r'C:\Keel\example-agency\data\keel.db'); c.execute('PRAGMA foreign_keys=ON'); n=c.execute('DELETE FROM documents WHERE source LIKE ?', ('%family-agreement-template.docx',)).rowcount; c.commit(); print(n, 'document(s) removed')"
```

Signed family agreements hold personal details. The pilot ingests the blank template only, tagged
`coordinator`; the owner decides in `needs-from-client.md` whether completed agreements join the
corpus later, and if they do, they are tagged `coordinator` and redacted first with the PII redactor
(`keel/safety/pii.py`, `redact()`), run as a script over the files before they enter the corpus
folder; a `--redact` flag on `keel ingest` is on the follow-up list.

Injection screening runs on every chunk at ingest. A chunk the screen flags is stored quarantined and
stays out of retrieval; the admin page lists it with the reason. A false positive is reviewed by
reading the source passage from the quarantine list; the Release action on that row clears the flag
and records who released it in the ledger.

## 4. Users and roles

Keel identifies each request by a user id and the ACL tags that user carries
(`keel/answer/types.py`, `User`). This overlay's `roles` map says which tags a role receives and
`users` says who holds which role. In the pilot build identity is self-asserted: the chat page offers
a user picker, the picked user's tags travel with the request into retrieval, and the appliance relies
on the machine's own Windows login plus the loopback bind for who reaches the page at all
(`docs/web.md`). The picker's table is `DEMO_USERS` in `keel/web/views.py`; at install it is set from
this overlay, one entry per user with the tags of their role, so a carer's question is answered from
carer, staff and public material only.

- Adding a person: add a line under `users` in the client's copy of `keel.yaml` with their `user_id`
  and `role`, add the same id with the role's tags to the picker's table, then restart the app
  (`stop.ps1`, `run.ps1`). Keep the overlay and the table in step; the overlay is the record.
- Changing what a role may read: change the role's tag list in both places, restart. Documents keep
  their tags.
- Removing a person: remove the line in both places, restart. Their past requests stay in the
  inference log and the ledger, which is the audit trail working as intended.
- Placeholder ids (`carer-1`, `coordinator-1`) are replaced with the agency's own ids on their
  machine. Real names stay out of this repository.
- Per-person login is a later release; the pilot's guard is that the app answers on loopback only, on
  a machine the agency controls.

## 5. How approvals work

`create_ticket` is a write tool. When the model decides to call it, the policy marks the call
`needs_approval` and the agent loop places it in the approval queue instead of running it
(`keel/agent/policy.py`, `keel/agent/approvals.py`). The user sees "queued for approval"; the action
has yet to happen. The owner opens the admin page, reads the tool and its arguments, and approves or
rejects. An approved call runs once, its result is stored on the queue row, and each transition
(pending, approved or rejected, executed) is written to the hash-chained ledger with who decided.
Tools outside `policy.yaml` `allowed_tools` are refused before any of this and logged.

The same queue is reachable from the shell on the appliance, which suits a quick check without the
browser:

```powershell
.venv\Scripts\python.exe -m keel.cli approvals list --status pending
.venv\Scripts\python.exe -m keel.cli approvals approve 3 --by owner
.venv\Scripts\python.exe -m keel.cli approvals reject 4 --by owner
```

## 6. Reading the admin page

Open http://127.0.0.1:8400/admin (the Admin link in the app header). It is open on loopback, which
is the only place the pilot listens; should the app ever bind beyond loopback, the routes ask for the
`X-Keel-Admin-Token` header matching `KEEL_ADMIN_TOKEN`. What it shows, top to bottom:

- Totals and settings: requests answered, refusals, tokens, the profile, the model, `min_relevance`
  and whether air-gap is on. A glance confirms the appliance runs the way the overlay says.
- Recent requests: who asked, the question, whether the answer was a refusal, latency and tokens; each
  row opens a detail page with the retrieved chunk ids after the ACL filter, the tool calls, the
  ledger rows and any approvals for that request. A refusal on a question the person should be able
  to answer usually means the document is tagged narrower than intended, or the question sits below
  `min_relevance` (the source holds nothing on it).
- Fourteen-day trend: requests, refusal rate, latency, and judge scores when an eval run has been
  recorded.
- Quarantine list: chunks the injection screen held back, with the reason and the source document,
  and the Release action.
- Approvals: the pending queue with Approve and Reject, and the last ten decided items.
- Ledger: Verify recomputes the hash chain and names the first broken link if there is one; Export
  downloads the chain as JSONL for an auditor. From the shell, `keel verify-ledger` does the same
  check; run it after any restore and once a week.

## 7. Backup

Everything Keel knows sits in one file, `C:\Keel\example-agency\data\keel.db` (with `-wal` and `-shm`
companions while the app runs). Hot backup while the app is running:

```powershell
New-Item -ItemType Directory -Force C:\Keel\example-agency\backups | Out-Null
.venv\Scripts\python.exe -c "import sqlite3, datetime; s=sqlite3.connect(r'C:\Keel\example-agency\data\keel.db'); d=sqlite3.connect(r'C:\Keel\example-agency\backups\keel-%s.db' % datetime.date.today()); s.backup(d); d.close(); print('backup written')"
```

Restore: stop the app, put the file back as `keel.db`, start, run `keel verify-ledger`. Documents are
re-ingestable from `corpus\` at any time, so the database and the `corpus\` folder together are the
whole backup set. Keep the backup folder on the same encrypted disk or an offline drive the owner
holds; it contains everything the corpus contains.

## 8. What to send Blake before the pilot

- The completed `needs-from-client.md` (documents by class, people by role, machine details).
- The owner's yes to the role table in section 3 and to the tool policy (`policy.yaml`).
- The name of the person who approves queue items and the person who runs the weekly backup.
- A screenshot of the fixture demo answering on their machine (step 2), which proves the hardware.

## 9. Troubleshooting

- llama-server exits at start: read `C:\Keel\example-agency\data\llama-server.err`; the usual causes are
  a wrong `KEEL_MODEL_PATH` or too little RAM for the model. `-TimeoutSeconds 600` helps a slow disk.
- The web app is up but answers fail with a model error, or http://127.0.0.1:8400/health reports the
  model unreachable: llama-server is down; `run.ps1 -SkipWeb` brings it back.
- Everything refuses: `KEEL_DATA_DIR` points at a directory other than the one that was ingested (the
  admin page totals show zero documents), or the picked user carries none of the documents' tags. In
  agent mode, also confirm `policy.yaml` in that directory lists `search_docs`.
- Air-gap violations in `keel-web.err`: something tried to reach the network. That is the guard doing
  its job; the message names the target so the cause can be found.
