# Standard Embedding Defaults To Volces

## Goal

Make the standard knowledge-base embedding path usable in the current environment without requiring `DASHSCOPE_API_KEY`.

After this change, standard knowledge bases should be able to:

- upload and chunk documents
- upsert vectors successfully
- answer RAG queries using the existing `VOLCES_API_KEY`

## Decision

Standard knowledge-base text embedding will default to Volces instead of DashScope.

DashScope will no longer be the default provider for standard embedding. It can remain as a compatibility path only if the code still needs it, but the default runtime path must succeed with the current `.env` where `VOLCES_API_KEY` is set and `DASHSCOPE_API_KEY` is empty.

## Scope

In scope:

- update the standard embedding service to use a Volces text-embedding endpoint by default
- align config and environment-variable naming with the new default behavior
- keep multimodal embedding behavior unchanged
- add regression tests for the standard embedding service path
- verify a real standard knowledge-base end-to-end smoke flow

Out of scope:

- redesigning the multimodal embedding flow
- changing the LLM chat provider
- refactoring unrelated RAG retrieval logic

## Approach

### 1. Provider behavior

`EmbeddingService` becomes Volces-first for standard text embeddings.

The service should:

- use the configured Volces base URL and API key
- call the correct Volces text embedding endpoint
- request the configured output dimension
- return vectors in the same shape expected by the rest of the Milvus pipeline

If a compatibility branch for DashScope is kept, it should be explicit rather than accidental. The default path must not depend on `DASHSCOPE_API_KEY`.

### 2. Config model

The current config still describes standard embedding as DashScope-based. That creates a mismatch between runtime defaults and the actual environment.

We will update:

- config comments and naming where needed so they describe Volces as the default standard embedding provider
- `.env.example` wording so a fresh setup points users at the working provider

We will avoid a broad config redesign unless the implementation clearly needs one.

### 3. Backward compatibility

The output contract of `EmbeddingService` must stay stable:

- `embed_texts()` returns `List[List[float]]`
- `embed_query()` returns `List[float]`

That keeps Milvus upsert and hybrid retrieval unchanged outside the provider swap.

### 4. Verification

We will verify at three layers:

- unit regression tests for the standard embedding service
- local service-level smoke checks against the live Volces API
- end-to-end standard KB smoke test: create KB, upload text, chunk, upsert, query, confirm expected answer

## Risks

### Volces endpoint mismatch

The biggest risk is calling the wrong text-embedding endpoint or parsing the wrong response shape. We will isolate that with a focused regression test and a direct live API check before relying on the full RAG path.

### Dimension mismatch

If Volces returns a different vector dimension than Milvus expects, upsert will fail. The implementation must explicitly pass the configured dimension and preserve the current Milvus collection expectations.

### Existing DashScope assumptions

Some comments, defaults, or error messages may still assume DashScope. We will update the high-signal ones that affect setup and runtime debugging, but avoid sprawling cleanup unrelated to behavior.

## Testing Plan

1. Add a failing regression test for the standard embedding service:
   - correct Volces endpoint
   - correct response parsing
   - configured dimension propagation
2. Implement the minimal provider change.
3. Run the regression tests.
4. Run a live standard embedding smoke call through the service.
5. Re-run the standard KB end-to-end smoke test and confirm the returned answer matches the uploaded document content.
