# Keel documentation

The top-level [`README.md`](../README.md) is the front door: what Keel is, the controls with the code
and tests behind each, the sixty-second demo, the architecture diagrams, the evaluation numbers, and
what is stubbed or unverified. The pages here go one level deeper. Every page is plain Markdown; every
command in them runs from the repository root with the project interpreter.

| Page | Read it when |
| --- | --- |
| [`architecture.md`](architecture.md) | You want the module-by-module walkthrough: the provider contracts, the SQLite data model, the request lifecycle for `/api/ask` and for the agent loop, and the ledger kinds. |
| [`cli.md`](cli.md) | You are at a shell. Every `keel` command, flag, output shape and exit code: `ingest`, `ask`, `agent`, `approvals`, `verify-ledger`, `status`, `serve`, `eval`, `export-log`. |
| [`web.md`](web.md) | You are in a browser. The chat page, citation chips, the source viewer with ACL enforcement, the JSON API, the admin page and the admin guard. |
| [`evals.md`](evals.md) | You want to know how quality is measured: the golden set, the metrics and their definitions, the LLM judge and its prompt, the regression gate, the HTML report, and the numbers observed. |
| [`onprem.md`](onprem.md) | You are deploying on one machine: the native runners, hardware notes, the model swap, air-gap mode and how to prove it, Docker Compose, backups and upgrades. |
| [`deploy-azure.md`](deploy-azure.md) | You are deploying to Azure. A short pointer; [`../deploy/azure/README.md`](../deploy/azure/README.md) is the authoritative guide. |
| [`threat-model.md`](threat-model.md) | You are reviewing security: assets, actors, trust boundaries, twelve threats with their controls, residual risks, and a command per control. [`../SECURITY.md`](../SECURITY.md) is the short public version. |
| [`demo-script.md`](demo-script.md) | You are recording or presenting the demo: a ninety-second script with what to type and what to point at, plus a block of questions to try. |

Related files outside this directory:

- [`security-review.md`](security-review.md): the pre-publish adversarial review, every finding with its severity and proving test.
- [`../CHANGELOG.md`](../CHANGELOG.md): release notes for 0.1.0, including the known limits.
- [`../deploy/aws/README.md`](../deploy/aws/README.md): the AWS stub and the mapping to Bedrock and OpenSearch Serverless.
- [`../fixtures/corpus.yaml`](../fixtures/corpus.yaml) and [`../fixtures/golden.yaml`](../fixtures/golden.yaml): the fixture corpus manifest and the golden question set the docs refer to.
