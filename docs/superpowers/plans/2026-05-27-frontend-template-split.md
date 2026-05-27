# Frontend Template Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前仓库内新增 `frontend-user/` 和 `frontend-admin/` 两个独立前端，完整承接现有用户端与管理端能力，并继续复用现有后端 `/api/v1/*` 接口，不改任何后端代码。

**Architecture:** `frontend-user/` 以模板工程为视觉基础，接入当前知识库问答、会话、历史和图片提问逻辑；`frontend-admin/` 使用同一视觉语言，承接管理员登录、知识库管理、数据导入、数据查看和服务记录/工单模块。旧 `frontend/` 作为迁移期间的对照与回退保留不动。

**Tech Stack:** Vue 3、TypeScript、Vite、Tailwind CSS v4、Axios、原有后端 `/api/v1/*` 接口、现有游客头与管理员鉴权逻辑。

---

## File Structure

- Create: `frontend-user/package.json`
- Create: `frontend-user/tsconfig.json`
- Create: `frontend-user/tsconfig.app.json`
- Create: `frontend-user/tsconfig.node.json`
- Create: `frontend-user/vite.config.ts`
- Create: `frontend-user/index.html`
- Create: `frontend-user/src/main.ts`
- Create: `frontend-user/src/style.css`
- Create: `frontend-user/src/App.vue`
- Create: `frontend-user/src/views/ChatView.vue`
- Create: `frontend-user/src/views/HistoryView.vue`
- Create: `frontend-user/src/components/AppHeader.vue`
- Create: `frontend-user/src/components/AppSidebar.vue`
- Create: `frontend-user/src/components/ChatArea.vue`
- Create: `frontend-user/src/components/WelcomeScreen.vue`
- Create: `frontend-user/src/components/InputArea.vue`
- Create: `frontend-user/src/components/SessionList.vue`
- Create: `frontend-user/src/components/MessageBubble.vue`
- Create: `frontend-user/src/components/SourcePanel.vue`
- Create: `frontend-user/src/components/QueryImageTray.vue`
- Create: `frontend-user/src/services/api.ts`
- Create: `frontend-user/src/services/auth.ts`
- Create: `frontend-user/src/utils/entryIdentity.ts`
- Create: `frontend-user/src/utils/conversationParams.ts`

- Create: `frontend-admin/package.json`
- Create: `frontend-admin/tsconfig.json`
- Create: `frontend-admin/tsconfig.app.json`
- Create: `frontend-admin/tsconfig.node.json`
- Create: `frontend-admin/vite.config.ts`
- Create: `frontend-admin/index.html`
- Create: `frontend-admin/src/main.ts`
- Create: `frontend-admin/src/style.css`
- Create: `frontend-admin/src/App.vue`
- Create: `frontend-admin/src/components/AdminShell.vue`
- Create: `frontend-admin/src/components/AdminHeader.vue`
- Create: `frontend-admin/src/components/AdminSidebar.vue`
- Create: `frontend-admin/src/components/AdminLoginDialog.vue`
- Create: `frontend-admin/src/components/AdminPageCard.vue`
- Create: `frontend-admin/src/components/AdminSectionHeader.vue`
- Create: `frontend-admin/src/views/CollectionsView.vue`
- Create: `frontend-admin/src/views/CreateCollectionView.vue`
- Create: `frontend-admin/src/views/ConfigView.vue`
- Create: `frontend-admin/src/views/DataImportView.vue`
- Create: `frontend-admin/src/views/DataViewView.vue`
- Create: `frontend-admin/src/views/ServiceTicketsView.vue`
- Create: `frontend-admin/src/services/api.ts`
- Create: `frontend-admin/src/services/auth.ts`
- Create: `frontend-admin/src/services/docApi.ts`
- Create: `frontend-admin/src/utils/adminNavigation.ts`
- Create: `frontend-admin/src/utils/adminConfig.ts`

- Copy then adapt into `frontend-admin/src/components/doc/`
  - `CategoryManager.vue`
  - `ChunkEditorPanel.vue`
  - `KnowledgeGraphPanel.vue`
  - `ExcelCategoryUpload.vue`
  - `DocUpload.vue`
  - `DocSearch.vue`
  - `DocList.vue`
  - `DocJobList.vue`

- Copy then adapt into `frontend-admin/src/components/admin/`
  - `AdminDataImport.vue`
  - `AdminDataView.vue`
  - `AdminServiceTickets.vue`

- Optional modify after validation only:
  - `README.md`

---

### Task 1: 搭建 `frontend-user` 工程骨架

**Files:**
- Create: `frontend-user/package.json`
- Create: `frontend-user/tsconfig.json`
- Create: `frontend-user/tsconfig.app.json`
- Create: `frontend-user/tsconfig.node.json`
- Create: `frontend-user/vite.config.ts`
- Create: `frontend-user/index.html`
- Create: `frontend-user/src/main.ts`
- Create: `frontend-user/src/style.css`
- Create: `frontend-user/src/App.vue`
- Create: `frontend-user/src/views/ChatView.vue`
- Create: `frontend-user/src/views/HistoryView.vue`

- [ ] **Step 1: 先写一个会失败的入口壳，确认新工程真实参与构建**

创建 `frontend-user/src/App.vue`，先引用尚未实现完整逻辑的页面组件：

```vue
<script setup lang="ts">
import { ref } from 'vue'
import ChatView from './views/ChatView.vue'
import HistoryView from './views/HistoryView.vue'

const activeView = ref<'chat' | 'history'>('chat')
</script>

<template>
  <div class="min-h-screen bg-background text-on-surface">
    <ChatView v-if="activeView === 'chat'" />
    <HistoryView v-else />
  </div>
</template>
```

- [ ] **Step 2: 运行构建，确认它因为缺少页面实现而失败**

Run:

```bash
cd frontend-user
pnpm install
pnpm build
```

Expected:

```text
FAIL
Could not resolve "./views/ChatView.vue"
```

- [ ] **Step 3: 补齐最小可构建骨架**

创建以下文件的最小实现：

`frontend-user/package.json`

```json
{
  "name": "rag-frontend-user",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5174",
    "build": "vue-tsc -b && vite build",
    "preview": "vite preview --port 4174"
  },
  "dependencies": {
    "axios": "^1.13.2",
    "lucide-vue-next": "^1.0.0",
    "vue": "^3.5.34"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.3.0",
    "@types/node": "^24.12.3",
    "@vitejs/plugin-vue": "^6.0.1",
    "@vue/tsconfig": "^0.9.1",
    "tailwindcss": "^4.3.0",
    "typescript": "~6.0.2",
    "vite": "^8.0.1",
    "vue-tsc": "^3.1.0"
  }
}
```

`frontend-user/tsconfig.json`

```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

`frontend-user/tsconfig.app.json`

```json
{
  "extends": "@vue/tsconfig/tsconfig.dom.json",
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"]
}
```

`frontend-user/tsconfig.node.json`

```json
{
  "extends": "@tsconfig/node22/tsconfig.json",
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo"
  },
  "include": ["vite.config.ts"]
}
```

`frontend-user/vite.config.ts`

```ts
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://localhost:8001',
        changeOrigin: true,
        proxyTimeout: 120000,
        timeout: 120000,
      },
    },
  },
})
```

`frontend-user/index.html`

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>扶商智能问答</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`frontend-user/src/main.ts`

```ts
import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

createApp(App).mount('#app')
```

`frontend-user/src/views/ChatView.vue`

```vue
<template>
  <section class="flex min-h-screen items-center justify-center">
    <div class="rounded-2xl bg-surface px-8 py-10 shadow-card">
      <h1 class="text-2xl font-semibold">用户问答前端</h1>
      <p class="mt-2 text-sm text-on-surface-variant">下一步接入真实问答能力</p>
    </div>
  </section>
</template>
```

`frontend-user/src/views/HistoryView.vue`

```vue
<template>
  <section class="flex min-h-screen items-center justify-center">
    <div class="rounded-2xl bg-surface px-8 py-10 shadow-card">
      <h1 class="text-2xl font-semibold">历史记录页</h1>
      <p class="mt-2 text-sm text-on-surface-variant">下一步接入真实历史能力</p>
    </div>
  </section>
</template>
```

`frontend-user/src/style.css`

```css
@import "tailwindcss";

@theme {
  --color-primary: #4f46e5;
  --color-primary-container: #e0e7ff;
  --color-on-primary: #ffffff;
  --color-background: #f9fafb;
  --color-surface: #ffffff;
  --color-surface-container: #f3f4f6;
  --color-surface-container-high: #e5e7eb;
  --color-on-surface: #111827;
  --color-on-surface-variant: #6b7280;
  --color-success: #10b981;
  --color-error: #ef4444;
  --color-outline-variant: #e5e7eb;
  --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.08);
}

body {
  margin: 0;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

- [ ] **Step 4: 运行构建，确认最小骨架通过**

Run:

```bash
cd frontend-user
pnpm build
```

Expected:

```text
vite v8...
✓ built in ...
```

- [ ] **Step 5: 提交这一小步**

```bash
git add frontend-user
git commit -m "feat: scaffold frontend-user app"
```

### Task 2: 接入 `frontend-user` 的真实问答接口与身份透传

**Files:**
- Create: `frontend-user/src/services/api.ts`
- Create: `frontend-user/src/services/auth.ts`
- Create: `frontend-user/src/utils/entryIdentity.ts`
- Create: `frontend-user/src/utils/conversationParams.ts`
- Modify: `frontend-user/src/views/ChatView.vue`
- Modify: `frontend-user/src/App.vue`

- [ ] **Step 1: 先让 `ChatView` 依赖真实 service，但暂时不实现，制造类型失败**

把 `frontend-user/src/views/ChatView.vue` 改成先引用尚未创建的服务：

```vue
<script setup lang="ts">
import { onMounted } from 'vue'
import { apiService } from '@/services/api'

onMounted(async () => {
  await apiService.listCollections()
})
</script>

<template>
  <div class="p-8">loading...</div>
</template>
```

- [ ] **Step 2: 运行构建，确认因为缺少 service 文件失败**

Run:

```bash
cd frontend-user
pnpm build
```

Expected:

```text
FAIL
Cannot find module '@/services/api'
```

- [ ] **Step 3: 落地用户侧接口层与身份工具**

创建 `frontend-user/src/utils/entryIdentity.ts`：

```ts
const GUEST_KEY = 'rag_guest_id'

export type EntryIdentity = {
  isStoreEntry: boolean
  userId: string
  nickname: string
  kb: string
}

const makeGuestId = () => {
  const raw = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `guest_${raw.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 48)}`
}

export const getGuestId = () => {
  let guestId = localStorage.getItem(GUEST_KEY)
  if (!guestId) {
    guestId = makeGuestId()
    localStorage.setItem(GUEST_KEY, guestId)
  }
  return guestId
}

export const getEntryIdentity = (): EntryIdentity => {
  const url = new URL(window.location.href)
  return {
    isStoreEntry: url.searchParams.get('store_entry') === '1',
    userId: url.searchParams.get('user_id') || '',
    nickname: url.searchParams.get('nickname') || '',
    kb: url.searchParams.get('kb') || '',
  }
}

export const buildEntryHeaders = () => {
  const entry = getEntryIdentity()
  const headers: Record<string, string> = {}
  if (entry.userId) headers['X-Entry-User-Id'] = entry.userId
  if (entry.nickname) headers['X-Entry-Nickname'] = entry.nickname
  if (entry.kb) headers['X-Entry-KB'] = entry.kb
  return headers
}
```

创建 `frontend-user/src/services/auth.ts`：

```ts
import { buildEntryHeaders, getGuestId } from '@/utils/entryIdentity'

export const attachGuestHeaders = (headers: Record<string, string> = {}) => ({
  ...headers,
  'X-Guest-Id': getGuestId(),
  ...buildEntryHeaders(),
})
```

创建 `frontend-user/src/services/api.ts`：

```ts
import axios from 'axios'
import { attachGuestHeaders } from './auth'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  config.headers = attachGuestHeaders(config.headers as Record<string, string>)
  return config
})

export const apiService = {
  async listCollections() {
    const response = await api.get('/admin/collections')
    return response.data
  },
  async knowledgeQueryStream(payload: Record<string, unknown>, handlers: Record<string, Function>, signal?: AbortSignal) {
    const res = await fetch('/api/v1/knowledge/stream', {
      method: 'POST',
      headers: attachGuestHeaders({
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'Accept-Encoding': 'identity',
      }),
      body: JSON.stringify(payload),
      signal,
    })
    if (!res.ok) throw new Error(`stream HTTP ${res.status}`)
    const reader = res.body?.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (reader) {
      const { done, value } = await reader.read()
      if (value) buffer += decoder.decode(value, { stream: !done })
      if (done) break
      const blocks = buffer.replace(/\r\n/g, '\n').split('\n\n')
      buffer = blocks.pop() || ''
      for (const block of blocks) {
        const event = block.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim()
        const data = block.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim()
        if (!data) continue
        const parsed = JSON.parse(data)
        if (event === 'meta') handlers.onMeta?.(parsed)
        if (event === 'delta') handlers.onDelta?.(parsed)
        if (event === 'done') handlers.onDone?.(parsed)
        if (event === 'error') handlers.onError?.(parsed)
      }
    }
  },
}
```

创建 `frontend-user/src/utils/conversationParams.ts`：

```ts
export const buildListSessionsParams = (kbName = '') => {
  const params: Record<string, string> = {}
  if (kbName.trim()) params.kb_name = kbName.trim()
  return params
}
```

- [ ] **Step 4: 用真实 service 改写 `ChatView` 的最小加载态**

更新 `frontend-user/src/views/ChatView.vue`：

```vue
<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { apiService } from '@/services/api'
import { getEntryIdentity } from '@/utils/entryIdentity'

const loading = ref(true)
const collections = ref<any[]>([])
const error = ref('')
const entryIdentity = getEntryIdentity()

onMounted(async () => {
  try {
    const res = await apiService.listCollections()
    collections.value = res.data?.collections || []
  } catch (err: any) {
    error.value = err?.message || '加载知识库失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section class="min-h-screen bg-background px-6 py-10">
    <div class="mx-auto max-w-4xl rounded-3xl bg-surface p-8 shadow-card">
      <h1 class="text-2xl font-semibold">扶商问答前端</h1>
      <p class="mt-2 text-sm text-on-surface-variant">
        {{ entryIdentity.isStoreEntry ? '门店入口模式' : '普通入口模式' }}
      </p>
      <p v-if="loading" class="mt-6 text-sm text-on-surface-variant">正在加载知识库...</p>
      <p v-else-if="error" class="mt-6 text-sm text-error">{{ error }}</p>
      <p v-else class="mt-6 text-sm text-on-surface-variant">已获取 {{ collections.length }} 个知识库</p>
    </div>
  </section>
</template>
```

- [ ] **Step 5: 验证接口层与构建**

Run:

```bash
cd frontend-user
pnpm build
pnpm dev
```

Expected:

```text
build 成功
本地打开 http://localhost:5174 后可看到加载知识库的页面骨架
浏览器网络面板出现 /api/v1/admin/collections 请求，并带有 X-Guest-Id 等头
```

### Task 3: 迁移 `frontend-user` 的真实聊天、会话和历史页

**Files:**
- Create: `frontend-user/src/components/AppHeader.vue`
- Create: `frontend-user/src/components/AppSidebar.vue`
- Create: `frontend-user/src/components/ChatArea.vue`
- Create: `frontend-user/src/components/WelcomeScreen.vue`
- Create: `frontend-user/src/components/InputArea.vue`
- Create: `frontend-user/src/components/SessionList.vue`
- Create: `frontend-user/src/components/MessageBubble.vue`
- Create: `frontend-user/src/components/SourcePanel.vue`
- Create: `frontend-user/src/components/QueryImageTray.vue`
- Modify: `frontend-user/src/services/api.ts`
- Modify: `frontend-user/src/views/ChatView.vue`
- Modify: `frontend-user/src/views/HistoryView.vue`
- Modify: `frontend-user/src/App.vue`

- [ ] **Step 1: 先给 `ChatView` 挂上真实组件引用，让构建先因为缺组件失败**

把 `frontend-user/src/views/ChatView.vue` 改成如下：

```vue
<script setup lang="ts">
import AppHeader from '@/components/AppHeader.vue'
import AppSidebar from '@/components/AppSidebar.vue'
import ChatArea from '@/components/ChatArea.vue'
import InputArea from '@/components/InputArea.vue'
</script>

<template>
  <div class="flex min-h-screen flex-col bg-background">
    <AppHeader />
    <div class="flex flex-1 min-h-0">
      <AppSidebar />
      <div class="flex flex-1 min-h-0 flex-col">
        <ChatArea />
        <InputArea />
      </div>
    </div>
  </div>
</template>
```

- [ ] **Step 2: 运行构建，确认缺少组件时报错**

Run:

```bash
cd frontend-user
pnpm build
```

Expected:

```text
FAIL
Could not resolve "@/components/AppHeader.vue"
```

- [ ] **Step 3: 迁移完整用户侧状态与组件**

此步将旧 `frontend/src/components/SimpleChat.vue` 和 `frontend/src/components/UserHistory.vue` 的逻辑拆入新结构，并补充以下接口到 `frontend-user/src/services/api.ts`：

```ts
async knowledgeQuery(query: string, sessionId = 'default', model: string | null = null, collection: string | null = null, forceMultiDoc: boolean | null = null, keywordFilter: string | null = null, queryImage: string | null = null, queryImages: string[] = []) {
  const payload: Record<string, unknown> = { query, session_id: sessionId }
  if (model) payload.model = model
  if (collection) payload.collection = collection
  if (forceMultiDoc != null) payload.force_multi_doc = forceMultiDoc
  if (keywordFilter) payload.keyword_filter = keywordFilter
  if (queryImage) payload.query_image = queryImage
  if (queryImages.length) payload.query_images = queryImages
  const response = await api.post('/knowledge', payload)
  return response.data
},
async listSessions(kbName = '') {
  const response = await api.get('/conversations', { params: buildListSessionsParams(kbName) })
  return response.data
},
async createSession(kbName: string, title = '新会话', userId = 'default') {
  const response = await api.post('/conversations', { kb_name: kbName, title, user_id: userId })
  return response.data
},
async getSessionMessages(sessionId: string, limit = 100) {
  const response = await api.get(`/conversations/${sessionId}/messages`, { params: { limit } })
  return response.data
},
async deleteSession(sessionId: string) {
  const response = await api.delete(`/conversations/${sessionId}`)
  return response.data
},
async resolveQueryImages(ossKeys: string[]) {
  const response = await api.post('/chunks/resolve-oss-keys', { oss_keys: ossKeys })
  return response.data
}
```

`frontend-user/src/App.vue` 目标形态：

```vue
<script setup lang="ts">
import { ref } from 'vue'
import ChatView from './views/ChatView.vue'
import HistoryView from './views/HistoryView.vue'

const activeView = ref<'chat' | 'history'>('chat')
</script>

<template>
  <component :is="activeView === 'chat' ? ChatView : HistoryView" @navigate="activeView = $event" />
</template>
```

`frontend-user/src/views/HistoryView.vue` 目标形态：

```vue
<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { apiService } from '@/services/api'

const records = ref<any[]>([])

onMounted(async () => {
  const res = await apiService.listCollections()
  records.value = res.data?.collections || []
})
</script>

<template>
  <section class="min-h-screen bg-background px-6 py-8">
    <div class="mx-auto max-w-5xl rounded-3xl bg-surface p-8 shadow-card">
      <h1 class="text-2xl font-semibold">历史记录</h1>
      <p class="mt-2 text-sm text-on-surface-variant">此页承接旧 UserHistory.vue 的恢复能力</p>
      <div class="mt-6 space-y-3">
        <div v-for="item in records" :key="item.name" class="rounded-2xl bg-surface-container px-4 py-3 text-sm">
          {{ item.display_name || item.name }}
        </div>
      </div>
    </div>
  </section>
</template>
```

工程实现时，页面最终必须完整承接：

- 知识库选择
- 会话创建/切换/删除
- 历史记录页恢复会话
- 图片上传预览与移除
- 流式回答
- 来源展示
- 门店/入口身份透传

- [ ] **Step 4: 运行用户侧完整人工验证**

Run:

```bash
cd frontend-user
pnpm build
pnpm dev
```

Expected:

```text
build 成功
用户端可以完成一次知识库问答
流式回答正常增量渲染
会话列表可以创建、切换、删除
历史页可以恢复会话
图片提问可以携带到接口请求中
```

- [ ] **Step 5: 提交用户端迁移**

```bash
git add frontend-user
git commit -m "feat: migrate user chat frontend"
```

### Task 4: 搭建 `frontend-admin` 工程骨架与登录壳

**Files:**
- Create: `frontend-admin/package.json`
- Create: `frontend-admin/tsconfig.json`
- Create: `frontend-admin/tsconfig.app.json`
- Create: `frontend-admin/tsconfig.node.json`
- Create: `frontend-admin/vite.config.ts`
- Create: `frontend-admin/index.html`
- Create: `frontend-admin/src/main.ts`
- Create: `frontend-admin/src/style.css`
- Create: `frontend-admin/src/App.vue`
- Create: `frontend-admin/src/components/AdminShell.vue`
- Create: `frontend-admin/src/components/AdminHeader.vue`
- Create: `frontend-admin/src/components/AdminSidebar.vue`
- Create: `frontend-admin/src/components/AdminLoginDialog.vue`
- Create: `frontend-admin/src/components/AdminPageCard.vue`
- Create: `frontend-admin/src/components/AdminSectionHeader.vue`
- Create: `frontend-admin/src/services/auth.ts`
- Create: `frontend-admin/src/services/api.ts`

- [ ] **Step 1: 先写一个依赖登录弹窗但文件不存在的壳**

创建 `frontend-admin/src/App.vue`：

```vue
<script setup lang="ts">
import AdminShell from './components/AdminShell.vue'
</script>

<template>
  <AdminShell />
</template>
```

创建 `frontend-admin/src/components/AdminShell.vue`：

```vue
<script setup lang="ts">
import AdminLoginDialog from './AdminLoginDialog.vue'
</script>

<template>
  <div>
    <AdminLoginDialog />
  </div>
</template>
```

- [ ] **Step 2: 运行构建，确认缺少登录弹窗时报错**

Run:

```bash
cd frontend-admin
pnpm install
pnpm build
```

Expected:

```text
FAIL
Could not resolve "./AdminLoginDialog.vue"
```

- [ ] **Step 3: 建立管理员工程与登录态基础**

`frontend-admin/package.json`

```json
{
  "name": "rag-frontend-admin",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5175",
    "build": "vue-tsc -b && vite build",
    "preview": "vite preview --port 4175"
  },
  "dependencies": {
    "axios": "^1.13.2",
    "lucide-vue-next": "^1.0.0",
    "vue": "^3.5.34"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.3.0",
    "@types/node": "^24.12.3",
    "@vitejs/plugin-vue": "^6.0.1",
    "@vue/tsconfig": "^0.9.1",
    "tailwindcss": "^4.3.0",
    "typescript": "~6.0.2",
    "vite": "^8.0.1",
    "vue-tsc": "^3.1.0"
  }
}
```

`frontend-admin/vite.config.ts`

```ts
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5175,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://localhost:8001',
        changeOrigin: true,
        proxyTimeout: 120000,
        timeout: 120000,
      },
    },
  },
})
```

`frontend-admin/index.html`

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>扶商后台管理</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`frontend-admin/src/main.ts`

```ts
import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

createApp(App).mount('#app')
```

`frontend-admin/src/style.css`

```css
@import "tailwindcss";

@theme {
  --color-primary: #4f46e5;
  --color-background: #f9fafb;
  --color-surface: #ffffff;
  --color-surface-container: #f3f4f6;
  --color-on-surface: #111827;
  --color-on-surface-variant: #6b7280;
  --color-error: #ef4444;
  --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.08);
}

body {
  margin: 0;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

`frontend-admin/src/services/auth.ts`

```ts
import axios from 'axios'

const TOKEN_KEY = 'rag_admin_token'
const USER_KEY = 'rag_admin_user'
export const AUTH_EXPIRED_EVENT = 'rag-auth-expired'

export const getAuthToken = () => localStorage.getItem(TOKEN_KEY) || ''

export const getStoredAdmin = () => {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export const setAuthSession = ({ token, user }: { token?: string; user?: unknown }) => {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export const clearAuthSession = () => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export const authService = {
  async login(username: string, password: string) {
    const response = await axios.post('/api/v1/auth/login', { username, password })
    const data = response.data?.data || {}
    setAuthSession({ token: data.access_token, user: data.user })
    return data.user
  },
  async me() {
    if (!getAuthToken()) return null
    const response = await axios.get('/api/v1/auth/me', {
      headers: { Authorization: `Bearer ${getAuthToken()}` },
    })
    return response.data?.data || null
  },
  async logout() {
    try {
      if (getAuthToken()) {
        await axios.post('/api/v1/auth/logout', null, {
          headers: { Authorization: `Bearer ${getAuthToken()}` },
        })
      }
    } finally {
      clearAuthSession()
    }
  },
}
```

`frontend-admin/src/components/AdminLoginDialog.vue`

```vue
<script setup lang="ts">
import { ref } from 'vue'
import { authService } from '@/services/auth'

const username = ref('admin')
const password = ref('')
const loading = ref(false)
const error = ref('')

const emit = defineEmits<{
  success: [user: any]
}>()

const submit = async () => {
  loading.value = true
  error.value = ''
  try {
    const user = await authService.login(username.value, password.value)
    emit('success', user)
  } catch (err: any) {
    error.value = err?.response?.data?.detail || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="mx-auto mt-24 max-w-md rounded-3xl bg-surface p-8 shadow-card">
    <h1 class="text-2xl font-semibold">管理员登录</h1>
    <div class="mt-6 space-y-4">
      <input v-model="username" class="w-full rounded-2xl bg-surface-container px-4 py-3 outline-none" placeholder="账号" />
      <input v-model="password" type="password" class="w-full rounded-2xl bg-surface-container px-4 py-3 outline-none" placeholder="密码" />
      <p v-if="error" class="text-sm text-error">{{ error }}</p>
      <button class="w-full rounded-2xl bg-primary px-4 py-3 text-white" :disabled="loading" @click="submit">
        {{ loading ? '登录中...' : '登录' }}
      </button>
    </div>
  </div>
</template>
```

- [ ] **Step 4: 验证管理员骨架与登录接口**

Run:

```bash
cd frontend-admin
pnpm build
pnpm dev
```

Expected:

```text
build 成功
打开 http://localhost:5175 后可以看到登录页
登录动作请求 /api/v1/auth/login
```

- [ ] **Step 5: 提交管理员工程骨架**

```bash
git add frontend-admin
git commit -m "feat: scaffold admin frontend app"
```

### Task 5: 迁移 `frontend-admin` 的知识库管理、导入和数据查看

**Files:**
- Create: `frontend-admin/src/services/docApi.ts`
- Create: `frontend-admin/src/utils/adminNavigation.ts`
- Create: `frontend-admin/src/utils/adminConfig.ts`
- Create: `frontend-admin/src/views/CollectionsView.vue`
- Create: `frontend-admin/src/views/CreateCollectionView.vue`
- Create: `frontend-admin/src/views/ConfigView.vue`
- Create: `frontend-admin/src/views/DataImportView.vue`
- Create: `frontend-admin/src/views/DataViewView.vue`
- Create: `frontend-admin/src/components/admin/AdminDataImport.vue`
- Create: `frontend-admin/src/components/admin/AdminDataView.vue`
- Create: `frontend-admin/src/components/doc/CategoryManager.vue`
- Create: `frontend-admin/src/components/doc/ChunkEditorPanel.vue`
- Create: `frontend-admin/src/components/doc/KnowledgeGraphPanel.vue`
- Create: `frontend-admin/src/components/doc/ExcelCategoryUpload.vue`
- Create: `frontend-admin/src/components/doc/DocUpload.vue`
- Create: `frontend-admin/src/components/doc/DocSearch.vue`
- Create: `frontend-admin/src/components/doc/DocList.vue`
- Create: `frontend-admin/src/components/doc/DocJobList.vue`
- Modify: `frontend-admin/src/components/AdminShell.vue`

- [ ] **Step 1: 先在 `AdminShell` 中引用数据导入页，让构建因页面缺失失败**

更新 `frontend-admin/src/components/AdminShell.vue`：

```vue
<script setup lang="ts">
import { ref } from 'vue'
import AdminHeader from './AdminHeader.vue'
import AdminSidebar from './AdminSidebar.vue'
import DataImportView from '@/views/DataImportView.vue'

const currentView = ref('data-import')
</script>

<template>
  <div class="flex min-h-screen bg-background">
    <AdminSidebar />
    <div class="flex flex-1 flex-col">
      <AdminHeader />
      <main class="flex-1 p-6">
        <DataImportView v-if="currentView === 'data-import'" />
      </main>
    </div>
  </div>
</template>
```

- [ ] **Step 2: 运行构建，确认缺少后台页面时报错**

Run:

```bash
cd frontend-admin
pnpm build
```

Expected:

```text
FAIL
Cannot find module '@/views/DataImportView.vue'
```

- [ ] **Step 3: 迁移后台基础页与文档 API**

将 `frontend/src/services/docApi.js` 按 TypeScript 语法迁入 `frontend-admin/src/services/docApi.ts`，保留全部现有接口名。

`frontend-admin/src/services/docApi.ts` 的开头应保持：

```ts
import axios from 'axios'

const BASE = '/api/v1'

export const docApi = {
  uploadDocument: (formData: FormData) => axios.post(`${BASE}/documents/upload`, formData),
  uploadDocumentToCategory: (formData: FormData) => axios.post(`${BASE}/documents/upload-to-category`, formData),
  batchUploadToCategory: (formData: FormData) => axios.post(`${BASE}/documents/batch-upload-to-category`, formData),
  searchDocuments: (formData: FormData) => axios.post(`${BASE}/documents/search`, formData),
  listJobs: (kbName: string, limit = 200) => axios.get(`${BASE}/jobs`, { params: { kb_name: kbName, limit } }),
  listCollections: () => axios.get(`${BASE}/admin/collections`),
  createCollection: (data: Record<string, unknown>) => axios.post(`${BASE}/admin/collections`, data),
  updateCollection: (kbName: string, data: Record<string, unknown>) => axios.put(`${BASE}/admin/collections/${kbName}`, data),
  deleteCollection: (kbName: string) => axios.delete(`${BASE}/admin/collections/${kbName}`),
}
```

后台视图页最小骨架：

`frontend-admin/src/views/DataImportView.vue`

```vue
<script setup lang="ts">
import AdminDataImport from '@/components/admin/AdminDataImport.vue'
</script>

<template>
  <AdminDataImport />
</template>
```

`frontend-admin/src/views/DataViewView.vue`

```vue
<script setup lang="ts">
import AdminDataView from '@/components/admin/AdminDataView.vue'
</script>

<template>
  <AdminDataView />
</template>
```

`frontend-admin/src/views/CollectionsView.vue`

```vue
<template>
  <section class="rounded-3xl bg-surface p-8 shadow-card">
    <h1 class="text-2xl font-semibold">知识库列表</h1>
  </section>
</template>
```

执行时，先直接复制旧文件，再做 import 与样式调整：

```powershell
Copy-Item 'frontend\src\components\AdminPanel.vue' 'frontend-admin\src\views\CollectionsView.vue'
Copy-Item 'frontend\src\components\admin\AdminDataImport.vue' 'frontend-admin\src\components\admin\AdminDataImport.vue'
Copy-Item 'frontend\src\components\admin\AdminDataView.vue' 'frontend-admin\src\components\admin\AdminDataView.vue'
Copy-Item 'frontend\src\components\doc\*' 'frontend-admin\src\components\doc\' -Recurse
```

复制后只允许做三类改动：

- 把旧相对路径 import 改成 `frontend-admin` 下的新路径
- 把页面容器、卡片、按钮、表格外壳改成模板风格
- 保持接口方法名、payload 和事件行为不变

- [ ] **Step 4: 验证管理员非工单模块**

Run:

```bash
cd frontend-admin
pnpm build
pnpm dev
```

Expected:

```text
build 成功
可以登录后台
知识库列表、创建、配置、数据导入、数据查看页面都能打开
相关网络请求继续命中原有 /api/v1/* 接口
```

- [ ] **Step 5: 提交后台基础业务迁移**

```bash
git add frontend-admin
git commit -m "feat: migrate admin knowledge management views"
```

### Task 6: 迁移 `frontend-admin` 的服务记录/工单模块

**Files:**
- Create: `frontend-admin/src/views/ServiceTicketsView.vue`
- Create: `frontend-admin/src/components/admin/AdminServiceTickets.vue`
- Modify: `frontend-admin/src/services/api.ts`
- Modify: `frontend-admin/src/components/AdminShell.vue`

- [ ] **Step 1: 先接入工单页引用，让缺失页面导致构建失败**

更新 `frontend-admin/src/components/AdminShell.vue`：

```vue
<script setup lang="ts">
import { ref } from 'vue'
import ServiceTicketsView from '@/views/ServiceTicketsView.vue'

const currentView = ref('tickets')
</script>

<template>
  <ServiceTicketsView v-if="currentView === 'tickets'" />
</template>
```

- [ ] **Step 2: 运行构建，确认缺少工单页时报错**

Run:

```bash
cd frontend-admin
pnpm build
```

Expected:

```text
FAIL
Cannot find module '@/views/ServiceTicketsView.vue'
```

- [ ] **Step 3: 迁移工单接口与页面**

在 `frontend-admin/src/services/api.ts` 中补齐以下接口：

```ts
import axios from 'axios'
import { getAuthToken } from './auth'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = getAuthToken()
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export const apiService = {
  async getServiceTickets(params: Record<string, unknown> = {}) {
    const response = await api.get('/admin/service-tickets', { params })
    return response.data
  },
  async getServiceTicketStats(params: Record<string, unknown> = {}) {
    const response = await api.get('/admin/service-tickets/stats', { params })
    return response.data
  },
  async getServiceTicket(id: string) {
    const response = await api.get(`/admin/service-tickets/${encodeURIComponent(id)}`)
    return response.data
  },
  async updateServiceTicket(id: string, payload: Record<string, unknown>) {
    const response = await api.patch(`/admin/service-tickets/${encodeURIComponent(id)}`, payload)
    return response.data
  },
  async deleteServiceTicket(id: string) {
    const response = await api.delete(`/admin/service-tickets/${encodeURIComponent(id)}`)
    return response.data
  },
  async updateServiceTicketChunk(id: string, chunkId: string, content: string) {
    const response = await api.put(`/admin/service-tickets/${encodeURIComponent(id)}/chunks/${encodeURIComponent(chunkId)}`, { content })
    return response.data
  },
  async revectorizeServiceTicket(id: string) {
    const response = await api.post(`/admin/service-tickets/${encodeURIComponent(id)}/revectorize`)
    return response.data
  },
}
```

`frontend-admin/src/views/ServiceTicketsView.vue`：

```vue
<script setup lang="ts">
import AdminServiceTickets from '@/components/admin/AdminServiceTickets.vue'
</script>

<template>
  <AdminServiceTickets />
</template>
```

执行时，将旧 `frontend/src/components/admin/AdminServiceTickets.vue` 迁入新项目并做样式改造，必须完整保留：

- 列表
- 筛选
- 统计
- 详情
- chat history
- sources
- chunk 修改
- 重新向量化
- 图片查看

推荐先复制旧文件，再只改表现层：

```powershell
Copy-Item 'frontend\src\components\admin\AdminServiceTickets.vue' 'frontend-admin\src\components\admin\AdminServiceTickets.vue'
```

复制后只允许做以下改动：

- 调整 import 到 `frontend-admin/src/services/api.ts`
- 用新后台壳包裹页面结构
- 把视觉层换成模板风格，但不改字段、方法和接口行为

- [ ] **Step 4: 运行工单模块联调验证**

Run:

```bash
cd frontend-admin
pnpm build
pnpm dev
```

Expected:

```text
build 成功
工单列表可以加载
工单详情可以打开
修改、删除、chunk 修补、重新向量化都继续命中旧接口
```

- [ ] **Step 5: 提交工单模块迁移**

```bash
git add frontend-admin
git commit -m "feat: migrate admin service tickets frontend"
```

### Task 7: 全量验证、回归和交付说明

**Files:**
- Modify: `README.md`（仅在确有必要时）

- [ ] **Step 1: 运行两个新前端的生产构建**

Run:

```bash
cd frontend-user
pnpm build

cd ../frontend-admin
pnpm build
```

Expected:

```text
两个前端都构建成功
```

- [ ] **Step 2: 运行两个新前端的本地联调**

Run:

```bash
cd frontend-user
pnpm dev

cd ../frontend-admin
pnpm dev
```

Expected:

```text
用户端 http://localhost:5174 可访问
管理端 http://localhost:5175 可访问
```

- [ ] **Step 3: 手工回归全部关键流程**

回归清单：

```text
用户端：
1. 知识库选择
2. 普通问题提问
3. 流式回答
4. 图片提问
5. 会话创建/切换/删除
6. 历史页恢复会话
7. 门店入口身份透传

管理端：
1. 登录/刷新/登出
2. 知识库列表/创建/配置
3. 数据导入
4. 数据查看
5. 服务记录/工单筛选与详情
6. 工单编辑/删除/chunk 修补/重新向量化
7. 文档与 chunk 子流程
```

- [ ] **Step 4: 如有必要再补充运行说明**

如果 `README.md` 需要补充，只增加最小运行指引：

```md
## Frontend Apps

- `frontend-user`: `pnpm install && pnpm dev`
- `frontend-admin`: `pnpm install && pnpm dev`
```

- [ ] **Step 5: 提交最终验证结果**

```bash
git add frontend-user frontend-admin README.md
git commit -m "feat: split rag frontends into user and admin apps"
```
