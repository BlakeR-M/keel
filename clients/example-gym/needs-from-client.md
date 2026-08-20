# Example Gym: what the gym provides before the pilot

Work through this with the owner. When every box is ticked, install day takes about an hour. Nothing
on this list is sent to Blake by email or uploaded anywhere: it stays on the gym's machine, and Blake
works with it there. The completed copy of this checklist (roles, counts, decisions) is the one thing
that comes back to Blake ahead of install day.

## 1. The machine

- [ ] One Windows 10 or 11 PC that stays at the gym (the front-desk PC is the natural home): 8 cores
      and 16 GB RAM (or a 12 GB NVIDIA card for the larger model), 20 GB free disk, a local
      administrator account for the install.
- [ ] Agreement that the appliance runs air-gapped: the app and model listen on that machine only.
- [ ] Windows login for each person who will use it (their own account, or the shared front-desk
      account with the app's user picker; the owner chooses).
- [ ] Disk encryption on (BitLocker), and an offline drive or encrypted folder for weekly backups.

## 2. Documents, sorted by class

Sorted into folders on the machine, under `C:\Keel\example-gym\corpus\<class>\`. Formats: PDF, DOCX,
Markdown, HTML or plain text. Current versions only; superseded copies stay out.

| Class | Folder | Read by | Have it | Count |
| --- | --- | --- | --- | --- |
| Membership terms, class schedule, gym rules | `public\` | everyone | [ ] | |
| Staff procedures, emergency and first aid plan, code of conduct | `staff\` | all staff | [ ] | |
| Opening and closing, point of sale and refunds, blank waiver, waiver handling | `front-desk\` | front desk and owner | [ ] | |
| Programming guide, class run sheets, equipment checks | `trainers\` | trainers and owner | [ ] | |
| Insurance summary, supplier contracts | `management\` | owner | [ ] | |

Decisions to record:

- [ ] Signed waivers and member records: in the pilot corpus, or out? Recommendation: out; the blank
      waiver and the handling procedure answer the procedural questions. If in: tagged `front-desk`
      and redacted with the PII redactor before ingest.
- [ ] Anything on the machine that must stay out of the corpus entirely (payroll, personal HR files,
      member payment details).
- [ ] Who owns document updates (adds the new term's class schedule to the folder and reruns ingest).

## 3. People and roles

Roles carry these tags: owner reads everything; front-desk reads `public`, `staff` and `front-desk`;
trainer reads `public`, `staff` and `trainer`.

| Role | How many people | Named on the machine (yes/no) |
| --- | --- | --- |
| owner | 1 | [ ] |
| front-desk | | [ ] |
| trainer | | [ ] |

- [ ] The owner confirms the role table above matches how the gym works.
- [ ] Any person who holds two roles (a trainer who also covers the desk): they get both tags, and
      the owner says so here.

## 4. Approvals and running it

- [ ] Who approves queue items (the `create_ticket` action and any write tool added later). Default:
      the owner.
- [ ] Who runs the weekly backup and the ledger check, and where the backup drive lives.
- [ ] Who Blake calls when the answer looks wrong (a citation that points at the wrong section, a
      refusal on a question the documents do answer).

## 5. Sign-off

- [ ] The owner has read `runbook.md` sections 3 to 7 and agrees with the role table and the tool policy.
- [ ] Install day and time agreed; the machine is on the desk with the documents in their folders.
