# Example Nannying Agency: what the agency provides before the pilot

Work through this with the owner. When every box is ticked, install day takes about an hour. Nothing
on this list is sent to Blake by email or uploaded anywhere: it stays on the agency's machine, and
Blake works with it there. The completed copy of this checklist (roles, counts, decisions) is the
one thing that comes back to Blake ahead of install day.

## 1. The machine

- [ ] One Windows 10 or 11 PC or laptop that stays at the agency: 8 cores and 16 GB RAM (or a 12 GB
      NVIDIA card for the larger model), 20 GB free disk, a local administrator account for the install.
- [ ] Agreement that the appliance runs air-gapped: the app and model listen on that machine only.
- [ ] Windows login for each person who will use it (their own account, or a shared kiosk account with
      the app's user picker; the owner chooses).
- [ ] Disk encryption on (BitLocker), and an offline drive or encrypted folder for weekly backups.

## 2. Documents, sorted by class

Sorted into folders on the machine, under `C:\Keel\example-agency\corpus\<class>\`. Formats: PDF, DOCX,
Markdown, HTML or plain text. Current versions only; superseded copies stay out.

| Class | Folder | Read by | Have it | Count |
| --- | --- | --- | --- | --- |
| Service guide, family FAQ | `public\` | everyone | [ ] | |
| Policies: child safety, code of conduct, WHS, privacy, complaints | `policies\` | all staff | [ ] | |
| Carer handbook, onboarding pack, shift procedures | `carers\` | carers and up | [ ] | |
| Family agreement blank template, placement and rostering procedure, incident handling | `coordination\` | coordinators and owner | [ ] | |
| Insurance summary, supplier and contractor agreements | `management\` | owner | [ ] | |

Decisions to record:

- [ ] Completed (signed) family agreements: in the pilot corpus, or out? Recommendation: out for the
      pilot; the blank template answers the procedural questions. If in: tagged `coordinator` and
      redacted with the PII redactor before ingest.
- [ ] Anything on the machine that must stay out of the corpus entirely (payroll, personal HR files).
- [ ] Who owns document updates (adds a revised policy to the folder and reruns ingest).

## 3. People and roles

Roles carry these tags: owner reads everything; coordinator reads `public`, `staff`, `carer` and
`coordinator`; carer reads `public`, `staff` and `carer`.

| Role | How many people | Named on the machine (yes/no) |
| --- | --- | --- |
| owner | 1 | [ ] |
| coordinator | | [ ] |
| carer | | [ ] |

- [ ] The owner confirms the role table above matches how the agency works.
- [ ] Any person who needs a role outside this table (for example a bookkeeper), and what they read.

## 4. Approvals and running it

- [ ] Who approves queue items (the `create_ticket` action and any write tool added later). Default:
      the owner.
- [ ] Who runs the weekly backup and the ledger check, and where the backup drive lives.
- [ ] Who Blake calls when the answer looks wrong (a citation that points at the wrong section, a
      refusal on a question the documents do answer).

## 5. Sign-off

- [ ] The owner has read `runbook.md` sections 3 to 7 and agrees with the role table and the tool policy.
- [ ] Install day and time agreed; the machine is on the desk with the documents in their folders.
