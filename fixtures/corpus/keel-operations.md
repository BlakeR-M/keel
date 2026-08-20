# Keel — Operations Notes (fixture)

Keel is a sovereign retrieval-augmented generation and agent appliance. This fixture describes how it is
operated so the demo corpus contains something about itself.

## Profiles

Keel runs in one of two profiles. The local profile uses a llama.cpp server on the same machine, a
fastembed embedding model on the CPU, and a SQLite store. The azure profile uses Azure OpenAI for
generation and embeddings, Azure AI Search for the vector index, and DefaultAzureCredential for
authentication, so no keys are stored anywhere.

## Air-gap mode

When the environment variable KEEL_AIRGAP is set to 1, Keel refuses every outbound network connection
except to 127.0.0.1. This is enforced in code and covered by a test.

## Ledger

Every request, retrieval, tool call and answer is written to a hash-chained ledger. The command
`keel verify-ledger` recomputes the chain and reports the first broken link, if any.

## Approvals

Tools marked as write tools are never executed unattended. Their calls are placed in an approval queue
and a person approves or rejects them from the admin page.
