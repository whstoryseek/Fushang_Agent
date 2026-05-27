# Frontend Template Split Design

## Goal

Replace the current mixed Q&A frontend with a new template-driven user frontend, build a separate admin frontend in the same visual language, and keep the backend API surface and backend feature logic unchanged.

The approved constraints are:

- keep all existing backend endpoints, request contracts, response contracts, and business logic untouched
- preserve all current user-side capabilities
- preserve all current admin-side capabilities
- place both new frontends inside the current repository
- keep the legacy `frontend/` as a temporary fallback during migration

## Existing Context

The repository currently mixes user and admin experiences inside a single Vue application under `frontend/`:

- `frontend/src/App.vue` mounts both the chat experience and all admin modules in one shell
- `frontend/src/components/SimpleChat.vue` owns the current user-side knowledge-base Q&A flow, session sidebar, image query support, and streaming answer rendering
- `frontend/src/components/UserHistory.vue` provides the user conversation history page
- `frontend/src/components/AdminPanel.vue`, `frontend/src/components/admin/AdminDataImport.vue`, `frontend/src/components/admin/AdminDataView.vue`, and `frontend/src/components/admin/AdminServiceTickets.vue` provide the current admin features
- `frontend/src/services/api.js` already captures the current `/api/v1/*` contracts for user Q&A and admin ticket/collection APIs
- `frontend/src/services/auth.js` already captures the admin auth lifecycle and the guest/header behavior required by the backend
- `frontend/src/utils/entryIdentity.mjs` and related helpers carry store-entry identity headers that the backend already understands

The provided template project at `C:/Users/story/Desktop/扶商智能体知识库/前端重构/projects` is a user-facing visual shell built with Vue 3, TypeScript, Vite, and Tailwind CSS v4. It currently contains UI-only mock components:

- `src/App.vue`
- `src/components/AppHeader.vue`
- `src/components/AppSidebar.vue`
- `src/components/ChatArea.vue`
- `src/components/WelcomeScreen.vue`
- `src/components/InputArea.vue`

The template defines the target visual language:

- restrained, high-whitespace chat layout
- indigo-primary palette on very light surfaces
- rounded cards and bubbles
- soft interaction states instead of heavy enterprise chrome

The admin frontend should follow that same style rather than defaulting to a dense traditional dashboard aesthetic.

## Recommended Approach

Create two new independent frontend applications in the current repository:

- `frontend-user/` for the end-user Q&A experience
- `frontend-admin/` for the administrator experience

Both applications should call the existing backend APIs directly under `/api/v1/*`. This is a frontend-only migration, not an API redesign or backend refactor.

The migration should prefer low-risk duplication over premature abstraction:

- each new frontend keeps its own service layer
- stable logic from the legacy frontend may be copied and adapted into the new applications where needed
- no shared package should be introduced in this phase unless it becomes necessary to preserve behavior exactly

This keeps the change focused on replacing the presentation layer while reducing risk from cross-project build tooling or package extraction.

## Target Architecture

### Application Split

`frontend-user/` is responsible for:

- knowledge-base Q&A
- streaming answer rendering
- session list and session recovery
- image-based question submission
- user conversation history
- store-entry identity propagation

`frontend-admin/` is responsible for:

- admin login and session persistence
- knowledge-base collection management
- collection creation and retrieval configuration
- data import workflows
- data view workflows
- service ticket browsing and handling
- document, chunk, category, and graph-related admin tools currently exposed through the legacy admin UI

The legacy `frontend/` remains in place during implementation as:

- an interface contract reference
- a behavior reference for parity checks
- a rollback option until both new frontends are validated

### User Frontend Structure

`frontend-user/` should be based on the provided template structure, expanded only where current behavior requires it. The intended structure is:

```text
frontend-user/
  src/
    App.vue
    main.ts
    style.css
    components/
      AppHeader.vue
      AppSidebar.vue
      ChatArea.vue
      WelcomeScreen.vue
      InputArea.vue
      SessionList.vue
      MessageBubble.vue
      SourcePanel.vue
      QueryImageTray.vue
    views/
      ChatView.vue
      HistoryView.vue
    services/
      api.ts
      auth.ts
    utils/
      entryIdentity.ts
      conversationParams.ts
```

The template components stay visually recognizable, but their state becomes real:

- `AppSidebar.vue` should evolve from mock history into the real knowledge session list and search affordance
- `ChatArea.vue` should render real conversation messages, stream updates, source blocks, confidence metadata, and images
- `InputArea.vue` should submit real text and image payloads, preserve enter/shift-enter behavior, and surface disabled states from the current product rules
- `WelcomeScreen.vue` should remain the first-run empty state and quick-prompt entry point

The user frontend should not contain admin login or admin navigation.

### Admin Frontend Structure

`frontend-admin/` should use the same visual language but a structure optimized for management flows:

```text
frontend-admin/
  src/
    App.vue
    main.ts
    style.css
    components/
      AdminShell.vue
      AdminHeader.vue
      AdminSidebar.vue
      AdminLoginDialog.vue
      AdminPageCard.vue
      AdminSectionHeader.vue
    views/
      CollectionsView.vue
      CreateCollectionView.vue
      ConfigView.vue
      DataImportView.vue
      DataViewView.vue
      ServiceTicketsView.vue
    components/
      doc/
      admin/
    services/
      api.ts
      auth.ts
    utils/
      adminNavigation.ts
      adminConfig.ts
```

The admin app should preserve all legacy admin capabilities, but reorganize them into a dedicated shell:

- a separate admin login flow
- a dedicated admin sidebar
- a top bar with system state and current admin identity
- page-level cards, drawers, tables, and forms styled to match the template’s softness and spacing

The admin UI should keep template aesthetics as the first priority:

- light surfaces
- restrained borders
- rounded cards
- minimal but clear state color usage
- no heavy dark dashboard treatment

## Capability Mapping

### User Capability Mapping

The user app should preserve and migrate the following legacy behaviors:

- `frontend/src/components/SimpleChat.vue`
  - knowledge-only chat mode
  - current knowledge collection selection behavior
  - SSE consumption from `/api/v1/knowledge/stream`
  - source list expansion
  - image query upload and preview
  - session creation, switching, deletion, and restoration
- `frontend/src/components/UserHistory.vue`
  - user-visible history browsing and resume behavior
- `frontend/src/services/api.js`
  - `knowledgeQuery`
  - `knowledgeQueryStream`
  - `listCollections`
  - image resolution helpers if the user UI depends on them
- `frontend/src/utils/entryIdentity.mjs`
  - guest/store entry identification and request header forwarding

Admin-only features currently embedded in the user shell should be removed from the user app.

### Admin Capability Mapping

The admin app should preserve and migrate the following legacy behaviors:

- `frontend/src/services/auth.js`
  - login
  - logout
  - `me`
  - token persistence
  - auth-expired handling
- `frontend/src/components/AdminPanel.vue`
  - knowledge-base list
  - knowledge-base creation
  - retrieval configuration
  - config information display
- `frontend/src/components/admin/AdminDataImport.vue`
  - document upload
  - Excel category upload
- `frontend/src/components/admin/AdminDataView.vue`
  - collection-scoped data inspection
  - chunk/graph/document views currently reachable from that flow
- `frontend/src/components/admin/AdminServiceTickets.vue`
  - service ticket list
  - filters, stats, detail panes, edit actions, chunk patching, and re-vectorization
- `frontend/src/components/doc/*`
  - all document-management child workflows used by the admin experience

No admin capability should be dropped in this migration.

## Data Flow and API Integration

Both new frontends must preserve backend contracts exactly.

### Shared integration rules

- keep the `/api/v1/*` base path unchanged
- keep the current request methods unchanged
- keep payload field names unchanged
- keep auth header behavior unchanged
- keep guest/store-entry header behavior unchanged
- keep SSE parsing behavior unchanged for streaming answers
- keep current tolerance for missing optional fields unchanged

### User-side request flow

For user chat:

- the user sends text-only or text-plus-image input
- the user app builds the same payload shape currently accepted by `/api/v1/knowledge` or `/api/v1/knowledge/stream`
- the streaming parser keeps the current `meta`, `delta`, `done`, and `error` handling contract
- the UI reflects incremental answer updates without changing server behavior

For user history and session flows:

- the new UI should reuse the current session behaviors already encoded in the legacy frontend
- if the current frontend stores or derives any session identifiers locally, that mechanism should be preserved unless the new UI can match the same behavior with simpler state handling

### Admin-side request flow

For admin access:

- login continues to call `/api/v1/auth/login`
- current-user continues to call `/api/v1/auth/me`
- logout continues to call `/api/v1/auth/logout`

For admin data flows:

- collections continue to use `/api/v1/admin/collections`
- service tickets continue to use the existing `/api/v1/admin/service-tickets*` endpoints
- document and chunk tools continue to use their existing upload/list/update/delete endpoints

The admin frontend may reorganize screens, but it must not reinterpret or reshape backend responses in a way that changes visible behavior incorrectly.

## Styling and Interaction Rules

The provided template defines the baseline style for both new applications. The migration should keep that style coherent across user and admin surfaces.

Required styling principles:

- visual hierarchy comes from spacing, contrast, and surface tint before strong borders
- the primary indigo accent remains the default action color
- backgrounds stay light and airy
- typography remains clean and calm
- interaction feedback stays subtle and polished

Admin-specific interpretation rules:

- tables should be softened with card containers and gentle row states
- forms should use generous spacing and simple field grouping
- detail panels and drawers should feel like part of the same chat-product family
- operational status colors should stay restrained and readable

## Error Handling

The migration must not hide backend errors, but it may present them more clearly.

User app error handling should preserve:

- stream failure handling
- request timeout handling
- empty collection or unavailable knowledge-base states
- image upload validation or reset behavior already enforced on the frontend

Admin app error handling should preserve:

- authentication expiry handling
- per-page load failure messaging
- mutation failure messaging for collection, ticket, and document actions
- non-blocking recovery where the legacy app already supports retry or refresh

Any frontend-only enhancement must stay presentation-level. It must not add new backend assumptions.

## Testing and Validation Strategy

The migration should be validated at three levels.

### Contract parity

Verify that:

- all old user requests have matching requests in `frontend-user/`
- all old admin requests have matching requests in `frontend-admin/`
- auth and guest headers remain present where previously required
- streaming parsing behavior remains functionally identical

### Feature parity

Verify user flows:

- knowledge-base question submission
- streaming answer rendering
- session switching
- session deletion
- history page navigation and resume
- image-based question submission
- store-entry identity behavior

Verify admin flows:

- login, reload, and logout
- collections listing and creation
- retrieval/config editing
- data import
- data view
- service ticket filtering, detail inspection, mutation, and re-vectorization
- document and chunk child workflows

### Build validation

Both new frontends must:

- install cleanly
- run in local dev mode against the current backend
- build successfully for production

The legacy `frontend/` should remain untouched enough to serve as a comparison reference until final acceptance.

## Scope Boundaries

Included in scope:

- creating `frontend-user/`
- creating `frontend-admin/`
- wiring both to the current backend
- migrating all current user and admin frontend capabilities
- adapting the provided template style for both apps

Explicitly out of scope:

- changing backend Python code
- changing backend endpoint shapes
- changing backend business logic
- introducing a shared frontend package just for cleanliness
- deleting the legacy `frontend/` during the first migration pass
- redesigning product requirements beyond the approved visual replacement and app split

## Success Criteria

The work is successful when:

- the repository contains two runnable independent frontend apps: `frontend-user/` and `frontend-admin/`
- the user app visually follows the provided template while preserving all current user-side functionality
- the admin app visually follows the same template language while preserving all current admin-side functionality
- both apps use the existing backend API without requiring backend code changes
- the legacy mixed frontend remains available as a fallback until the new apps are accepted
