# Service Ticket Intent Flow

## Goal

Refactor the current ticket collection flow so the system can reliably route only the right requests into manual-work tickets while leaving normal RAG answers unchanged.

The approved classification boundary is:

- A: user asks the system or staff to perform a concrete action on a concrete business object, such as "帮我绑定门店" or "给他开通权限".
- B: RAG has already tried the normal knowledge-base flow and still cannot find a usable answer.
- C: the user message is too vague, incomplete, or image-only and cannot yet be classified safely.
- D: normal knowledge-base question. This includes tutorial, capability, and consultation phrasing such as "怎么绑定", "如何开通", "能不能关闭", and "XX 有什么用".

## Existing Context

The current code already has most of the infrastructure needed:

- `backend/app/api/v1/knowledge.py` is the request entry point for normal and streaming knowledge-base Q&A.
- `backend/app/services/knowledge_service.py` invokes the LangGraph RAG flow and persists conversation messages.
- `backend/app/services/service_ticket_service.py` owns ticket status decisions, operation clarification, collected fields, and ticket persistence.
- `backend/app/db/service_ticket_repository.py` stores tickets in `service_ticket` and retrieval snapshots in `service_ticket_context`.
- `service_ticket.clarification` is a JSON payload that can hold multi-turn state without a schema migration for the first implementation.

The design should extend these existing modules rather than create a second ticket system.

## Recommended Approach

Use a unified special-intent state machine inside `service_ticket_service.py`, with `knowledge.py` acting as a thin orchestration layer.

The first version should keep all durable state in the existing `service_ticket` row:

- `status`: `clarifying`, `pending_manual`, `resolved_ai`, `in_progress`, `resolved_manual`, or `ignored`
- `clarification_round`: current collection round
- `clarification`: structured JSON for intent class, field collection, turn history, retrieval result, image markers, and exit reason
- `service_ticket_context`: retrieval chunks used for SOP lookup or RAG fallback context

This keeps the change small, preserves the admin ticket UI/API, and avoids introducing a new table while the state model is still evolving.

## Intent Routing

Routing should happen in three places.

### Active Ticket Recovery

At the start of both `knowledge_qa` and `knowledge_qa_stream`, the API checks for an active `clarifying` ticket for the same user/session/knowledge base.

If an active ticket exists:

- merge the new user turn into the existing `clarification` payload
- respect refusal or "I do not know" replies
- decide whether to ask again, retry retrieval, finalize the ticket, or release the message back to normal RAG

This supports interruption recovery from any collection round.

### Pre-RAG Special Intent Detection

Only A and C should be intercepted before RAG.

A class must satisfy both conditions:

- concrete action target: account, store, employee, permission, payment account, order, page, or similar object
- delegation intent: "帮我", "给我", "替我", "处理一下", "弄一下", "安排一下", "开通/关闭/绑定/解绑/重置" used as a request to perform the action

The guardrail is strict: tutorial, consultation, reason, and capability questions stay D and go to RAG.

C class is for messages that are too short, incomplete, image-only, or dependent on missing context. It asks for clarification before the system tries to decide whether the issue is A, B, or D.

### Post-RAG B Detection

B class should not be guessed before retrieval. It should trigger only when normal RAG completes with no usable answer, such as:

- `used_fallback=True`
- `fallback_reason` indicates no relevant knowledge
- `quality_passed=False` because the answer says the knowledge base has no relevant content
- confidence is below the manual-review threshold and sources are weak

When B is triggered, the user-facing answer becomes a clarification question instead of a final low-confidence answer. The original RAG output and sources are stored in the ticket payload.

## A-Class Flow

For A class, the system first performs a silent SOP lookup through the existing retrieval preparation path.

If SOP is found:

- extract required fields from the retrieved SOP with the LLM
- normalize fields to stable keys such as `issue_detail`, `phone`, `image`, `id_card`, `store`, `account`, and `permission`
- ask only for missing fields
- after each user reply, merge extracted values and check completeness
- finalize to `pending_manual` once all required fields are present

If SOP is not found:

- create or update the ticket immediately as `pending_manual`
- mark `workflow_found=false`
- mark the exit reason as `no_standard_workflow`
- send the user the standard ticket-created message

This matches the requirement that A without SOP goes directly to backend processing.

## B-Class Flow

B class uses deep questioning for up to five rounds.

Round strategy:

- rounds 1-2: clarify phenomenon, scenario, exact error, and reproduction path
- rounds 3-4: confirm business background, affected store/account/user, impact scope, and urgency
- round 5: confirm the core issue and attempt a final category label

After each B-class user reply:

- merge the new information into `collected`
- build a richer query from the original question plus collected facts
- retry silent RAG retrieval
- if retrieval now produces a usable answer, exit the ticket flow and return the normal RAG answer as D
- if retrieval still misses after round 5, finalize to `pending_manual`

For image-only B candidates:

- run the existing image-aware model path when an image URL is available
- store only image metadata and recognized issue categories in the ticket payload
- ask the user to confirm one to three possible categories
- if the user confirms, route to A, B, or D based on the category
- if the user denies or cannot confirm, continue B questioning until the round cap

## C-Class Flow

C class asks clarification questions for up to three rounds.

The first C question should be direct and short, for example:

- "您的问题是否涉及某个具体业务场景或系统页面？"
- "能否补充具体操作步骤、报错信息，或说明截图中想处理的问题？"

After every C reply, rerun intent routing:

- if it becomes D, return to normal RAG
- if it becomes A, enter A-class flow
- if it remains unclear, ask the next C question
- if it is still unclear after three rounds, finalize to `pending_manual` with `exit_reason=semantic_unresolved`

## User Refusal

If the user explicitly says they do not want to answer, do not know, cannot provide, or wants staff to handle it directly:

- stop further questioning
- finalize the active ticket as `pending_manual`
- set `completion_status=incomplete`
- set `exit_reason=user_refused_or_unknown`
- keep all collected information and turn history

## Ticket Payload

The `clarification` JSON should use a stable structure:

```json
{
  "intent_class": "A|B|C",
  "reason": "operation_required|knowledge_missing|ambiguous_query",
  "original_query": "...",
  "required_fields": ["issue_detail", "phone", "image"],
  "missing_fields": ["phone"],
  "collected": {
    "issue_detail": "...",
    "phone": "...",
    "image_keys": ["..."],
    "has_image": true
  },
  "workflow_found": true,
  "workflow_summary": "...",
  "workflow_sources": [],
  "kb_result": {
    "hit": false,
    "fallback_reason": "no_relevant_knowledge",
    "confidence": 0.21
  },
  "image_analysis": {
    "has_image": true,
    "categories": ["门店绑定", "权限开通"]
  },
  "turns": [
    {
      "round": 1,
      "query": "...",
      "answer": "...",
      "status": "clarifying"
    }
  ],
  "completion_status": "complete|incomplete",
  "exit_reason": "fields_complete|no_standard_workflow|max_rounds|semantic_unresolved|user_refused_or_unknown|rag_hit_after_clarification",
  "ready_for_manual": true
}
```

The implementation may keep existing keys for backward compatibility, but new logic should write these canonical keys.

## User-Facing Exit

When the flow finalizes a manual ticket, the response should use the required ending shape:

`已为您记录问题，工单号 {ticket_id}，后续将由专人处理。`

If information is incomplete, add one short sentence before or after that message explaining that the ticket was created with currently available information.

After finalization, the active `clarifying` state ends because the ticket status becomes `pending_manual`.

## Error Handling

LLM classifier failure should default to D, unless deterministic rules clearly identify C. This prevents over-routing normal questions into manual tickets.

SOP retrieval or SOP extraction failure should not block the user. For A class, it should finalize a ticket with `workflow_found=false` and `fallback_used=true`.

B-class retry retrieval failure counts as a miss for that turn and stores the error message under `kb_result.error`.

Image analysis failure should ask a textual clarification question and store `image_analysis.error` without exposing internal errors to the user.

Ticket persistence failure should be logged and should not crash normal RAG for D-class questions. For active A/B/C ticket flows, persistence failure should return a user-friendly error because state recovery depends on the ticket row.

## Testing Plan

Backend service tests:

- A tutorials such as "富友账户怎么绑定" remain D and do not start tickets.
- A delegation requests such as "帮我绑定富友账户" start A-class collection.
- A with SOP found asks only missing SOP fields.
- A with no SOP finalizes immediately to `pending_manual`.
- B starts only after RAG fallback, not before retrieval.
- B retries retrieval after user clarification and exits to D when retrieval hits.
- B finalizes after five misses.
- C short or image-only messages ask clarification and reroute after new information.
- C finalizes after three unclear rounds.
- refusal finalizes immediately with incomplete status.
- active `clarifying` tickets resume by session/user/kb.

API tests:

- non-stream and stream endpoints produce equivalent `finish_reason`, `thoughts.clarification`, and final ticket messages.
- D-class responses still return the original RAG answer, sources, confidence, and image map.

Repository tests:

- existing `service_ticket` fields continue to normalize correctly.
- `clarification` JSON round-trips canonical keys.

## Rollout Notes

This should be implemented behind the existing knowledge API without changing the request contract.

The admin UI can initially display the existing ticket fields and raw clarification JSON. A later UI pass can add dedicated columns for intent class, missing fields, and completion status.
