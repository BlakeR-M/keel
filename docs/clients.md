# Client pilots

Keel is deployed for real businesses through overlays: per-client configuration in `clients/<id>/`
that sits on top of an unchanged codebase. `clients/README.md` describes the overlay format and how a
third one is created. This page is the pilot playbook: the sequence from first conversation to a
running appliance, and what evidence is collected along the way.

## Pilots

| Client | Overlay | Runbook | Profile |
| --- | --- | --- | --- |
| Example Nannying Agency (nannying agency) | `clients/example-agency/` | `clients/example-agency/runbook.md` | local, air-gapped, one machine |
| Example Gym (gym) | `clients/example-gym/` | `clients/example-gym/runbook.md` | local, air-gapped, one machine |

## The sequence

1. **Scope.** Fill in `needs-from-client.md` with the owner: machine, documents by class, people by
   role, who approves, who backs up. The role table is agreed here; it becomes the `roles` block in
   `keel.yaml` and the `acl_tags` in the manifest.
2. **Prove the hardware.** `.\demo.ps1 -Airgap` on their machine with the fixture corpus. A cited
   answer on their box is the go signal for install day.
3. **Install.** Follow the runbook: environment from `keel.yaml` `settings`, `policy.yaml` copied into
   the data directory, documents ingested from the filled-in manifest, web app on loopback.
4. **Verify the controls on their data.** A user from the narrowest role asks a question that only a
   wider role's document answers and receives a refusal; the owner asks the same and receives an
   answer with citations; `keel verify-ledger` reports the chain intact; the admin page shows the
   requests. Record the three results in the install notes.
5. **Hand over.** The owner knows the admin page, the approval queue and the backup command. The
   completed checklist and the install notes are the pilot record.
6. **Review after two weeks.** Read the inference log with the owner: refusals on answerable questions
   point at tags set too narrow or documents that are missing; low-relevance answers point at the
   corpus needing a document. Adjust the manifest and re-ingest.

## Where the boundaries sit

- Client documents, exports, names and contact details stay on the client's machine. The repository
  holds templates and placeholders only.
- The tool policy is per client and lives at `<data_dir>/policy.yaml` on their machine, so a policy
  change is a file edit and a restart with nothing to redeploy.
- Both pilots run the local profile with `KEEL_AIRGAP=1`. The Azure profile (`deploy/azure/README.md`)
  is the path for a client who wants a hosted appliance; the overlay format is the
  same, with `settings.profile: azure` and the Azure endpoints in `settings`.
