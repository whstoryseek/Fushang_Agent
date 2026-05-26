<template>
  <div class="history-page">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{{ sessions.length }}</div>
        <div class="stat-label">历史会话</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ totalMessages }}</div>
        <div class="stat-label">累计消息</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ uniqueKbCount }}</div>
        <div class="stat-label">涉及知识库</div>
      </div>
    </div>

    <div class="toolbar">
      <el-select v-model="kbFilter" size="small" style="width: 220px" @change="loadSessions">
        <el-option label="全部知识库" value="" />
        <el-option
          v-for="item in collections"
          :key="item.name"
          :label="item.display_name || item.name"
          :value="item.name"
        />
      </el-select>
      <el-input
        v-model="keyword"
        size="small"
        clearable
        placeholder="搜索会话标题或消息内容"
        style="width: 280px"
      />
      <el-button size="small" @click="loadSessions" :loading="loading">刷新</el-button>
    </div>

    <div class="history-layout">
      <div class="session-panel">
        <div class="panel-title">会话列表</div>
        <div v-if="loading" class="panel-empty">正在加载历史会话...</div>
        <div v-else-if="filteredSessions.length === 0" class="panel-empty">暂无历史会话</div>
        <button
          v-for="session in filteredSessions"
          :key="session.id"
          class="session-card"
          :class="{ active: selectedSession?.id === session.id }"
          @click="selectSession(session)"
        >
          <div class="session-row">
            <div class="session-title">{{ session.title || '新会话' }}</div>
            <div class="session-time">{{ formatSessionTime(session.updated_at) }}</div>
          </div>
          <div class="session-kb">{{ session.kb_name }}</div>
          <div class="session-preview">
            {{ session.last_message_preview || '暂无消息内容' }}
          </div>
          <div class="session-meta">
            <span>{{ session.message_count }} 条消息</span>
            <span>{{ session.last_message_role === 'assistant' ? '最近回复' : '最近提问' }}</span>
          </div>
        </button>
      </div>

      <div class="detail-panel">
        <template v-if="selectedSession">
          <div class="detail-header">
            <div>
              <div class="detail-title">{{ selectedSession.title || '新会话' }}</div>
              <div class="detail-subtitle">
                <span>{{ selectedSession.kb_name }}</span>
                <span>{{ selectedSession.message_count }} 条消息</span>
                <span>{{ formatDateTime(selectedSession.updated_at) }}</span>
              </div>
            </div>
            <div class="detail-actions">
              <el-button size="small" type="primary" @click="resumeSession(selectedSession)">
                继续对话
              </el-button>
              <el-button size="small" type="danger" plain @click="deleteSession(selectedSession)">
                删除会话
              </el-button>
            </div>
          </div>

          <div v-if="detailLoading" class="panel-empty">正在加载消息详情...</div>
          <div v-else-if="sessionMessages.length === 0" class="panel-empty">这个会话还没有消息记录</div>
          <div v-else class="message-list">
            <div
              v-for="msg in sessionMessages"
              :key="msg.id"
              class="message-row"
              :class="msg.role"
            >
              <div class="message-badge">{{ msg.role === 'assistant' ? 'AI' : '我' }}</div>
              <div class="message-body">
                <div class="message-bubble" :class="msg.role">
                  <img
                    v-if="msg.queryImageUrl"
                    :src="msg.queryImageUrl"
                    class="query-image"
                  />
                  <div v-if="msg.isHtml" v-html="msg.content" />
                  <div v-else class="plain-text">{{ msg.content }}</div>
                </div>
                <div class="message-meta-line">
                  <span>{{ formatDateTime(msg.timestamp) }}</span>
                  <span v-if="msg.sources?.length">来源 {{ msg.sources.length }}</span>
                  <span v-if="msg.confidence != null">置信度 {{ Math.round(msg.confidence * 100) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </template>

        <div v-else class="panel-empty detail-empty">
          请选择左侧会话查看历史消息
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import MarkdownIt from 'markdown-it'

import { docApi } from '../services/docApi'

const md = new MarkdownIt({ html: false, linkify: true, typographer: true, breaks: true })

const loading = ref(false)
const detailLoading = ref(false)
const sessions = ref([])
const collections = ref([])
const sessionMessages = ref([])
const selectedSession = ref(null)
const kbFilter = ref('')
const keyword = ref('')

const IMAGE_CACHE_TTL = 50 * 60 * 1000

const totalMessages = computed(() => sessions.value.reduce((sum, item) => sum + (item.message_count || 0), 0))
const uniqueKbCount = computed(() => new Set(sessions.value.map(item => item.kb_name).filter(Boolean)).size)
const filteredSessions = computed(() => {
  const text = keyword.value.trim().toLowerCase()
  if (!text) return sessions.value
  return sessions.value.filter((item) => {
    const haystacks = [
      item.title || '',
      item.kb_name || '',
      item.last_message_preview || '',
    ]
    return haystacks.some(value => value.toLowerCase().includes(text))
  })
})

const getCachedUrl = (placeholder) => {
  try {
    const raw = sessionStorage.getItem(`img_cache:${placeholder}`)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (Date.now() > parsed.expiresAt) {
      sessionStorage.removeItem(`img_cache:${placeholder}`)
      return null
    }
    return parsed.url
  } catch {
    return null
  }
}

const setCachedUrl = (placeholder, url) => {
  try {
    sessionStorage.setItem(
      `img_cache:${placeholder}`,
      JSON.stringify({ url, expiresAt: Date.now() + IMAGE_CACHE_TTL }),
    )
  } catch {}
}

const sanitizeImagePlaceholders = (text, imageMap) => {
  if (!text) return ''
  if (!imageMap || !Object.keys(imageMap).length) return text
  const valid = new Set(Object.keys(imageMap))
  return text.replace(/<<IMAGE:[0-9a-fA-F]{8}>>/g, (placeholder) => (valid.has(placeholder) ? placeholder : ''))
}

const toMarkdownWithImages = (raw, imageMap) => {
  let content = sanitizeImagePlaceholders(raw, imageMap)
  Object.entries(imageMap || {}).forEach(([placeholder, url]) => {
    content = content.split(placeholder).join(`\n![image](${url})\n`)
  })
  return content
}

const formatSessionTime = (value) => {
  if (!value) return ''
  const date = new Date(value)
  const diff = Date.now() - date.getTime()
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  return date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}

const formatDateTime = (value) => {
  if (!value) return ''
  return new Date(value).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const loadCollections = async () => {
  try {
    const res = await docApi.listPublicKnowledgeBases()
    collections.value = res.data.data?.collections || []
  } catch (e) {
    console.warn('加载知识库列表失败:', e)
  }
}

const loadSessions = async () => {
  loading.value = true
  try {
    const res = await docApi.listSessions(kbFilter.value || null)
    sessions.value = res.data.data?.sessions || []
    if (selectedSession.value) {
      const nextSelected = sessions.value.find(item => item.id === selectedSession.value.id)
      if (nextSelected) {
        selectedSession.value = nextSelected
      } else {
        selectedSession.value = null
        sessionMessages.value = []
      }
    }
    if (!selectedSession.value && sessions.value.length > 0) {
      await selectSession(sessions.value[0])
    }
  } catch (e) {
    ElMessage.error('加载历史会话失败')
    console.warn('加载历史会话失败:', e)
  } finally {
    loading.value = false
  }
}

const selectSession = async (session) => {
  if (!session?.id) return
  selectedSession.value = session
  detailLoading.value = true
  try {
    const res = await docApi.getSessionMessages(session.id)
    const historyMessages = res.data.data?.messages || []

    const placeholders = [...new Set(historyMessages.flatMap(item => item.image_placeholders || []))]
    const missingPlaceholders = []
    const imageMap = {}
    for (const placeholder of placeholders) {
      const cached = getCachedUrl(placeholder)
      if (cached) imageMap[placeholder] = cached
      else missingPlaceholders.push(placeholder)
    }
    if (missingPlaceholders.length) {
      try {
        const imgRes = await docApi.resolveImages(missingPlaceholders)
        const freshMap = imgRes.data.data || {}
        Object.entries(freshMap).forEach(([placeholder, url]) => {
          imageMap[placeholder] = url
          setCachedUrl(placeholder, url)
        })
      } catch {}
    }

    const queryImageKeys = [...new Set(
      historyMessages
        .filter(item => item.role === 'user' && item.query_image_oss_key)
        .map(item => item.query_image_oss_key),
    )]
    let queryImageMap = {}
    if (queryImageKeys.length) {
      try {
        const queryImgRes = await docApi.resolveQueryImages(queryImageKeys)
        queryImageMap = queryImgRes.data.data || {}
      } catch {}
    }

    sessionMessages.value = historyMessages.map((item) => {
      const timestamp = new Date(item.created_at)
      if (item.role === 'assistant') {
        return {
          id: item.id,
          role: item.role,
          isHtml: true,
          content: md.render(toMarkdownWithImages(item.content || '', imageMap)),
          confidence: item.confidence,
          sources: item.sources || [],
          timestamp,
        }
      }

      const queryImageUrl = item.query_image_oss_key ? queryImageMap[item.query_image_oss_key] : ''
      return {
        id: item.id,
        role: item.role,
        isHtml: false,
        content: item.content || '',
        queryImageUrl,
        confidence: item.confidence,
        sources: item.sources || [],
        timestamp,
      }
    })
  } catch (e) {
    sessionMessages.value = []
    ElMessage.error('加载历史消息失败')
    console.warn('加载历史消息失败:', e)
  } finally {
    detailLoading.value = false
  }
}

const resumeSession = (session) => {
  window.dispatchEvent(new CustomEvent('knowledge-session:resume', {
    detail: {
      sessionId: session.id,
      kbName: session.kb_name,
    },
  }))
  ElMessage.success('已切换到知识问答并恢复该会话')
}

const deleteSession = async (session) => {
  try {
    await ElMessageBox.confirm(
      `确认删除会话“${session.title || '新会话'}”吗？`,
      '删除会话',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )
  } catch {
    return
  }

  try {
    await docApi.deleteSession(session.id)
    if (selectedSession.value?.id === session.id) {
      selectedSession.value = null
      sessionMessages.value = []
    }
    await loadSessions()
    ElMessage.success('会话已删除')
  } catch (e) {
    ElMessage.error('删除会话失败')
    console.warn('删除会话失败:', e)
  }
}

onMounted(async () => {
  await Promise.all([loadCollections(), loadSessions()])
})
</script>

<style scoped>
.history-page {
  max-width: 1280px;
  width: 100%;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 18px;
}

.stat-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 20px 22px;
}

.stat-value {
  font-size: 28px;
  line-height: 1;
  font-weight: 700;
  color: #f4f7ff;
}

.stat-label {
  margin-top: 8px;
  font-size: 12px;
  color: rgba(255,255,255,0.36);
}

.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 18px;
}

.toolbar :deep(.el-select),
.toolbar :deep(.el-input) {
  max-width: 100%;
}

.history-layout {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 16px;
  min-height: 620px;
}

.session-panel,
.detail-panel {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  overflow: hidden;
}

.session-panel {
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: calc(100vh - 240px);
  overflow-y: auto;
}

.detail-panel {
  display: flex;
  flex-direction: column;
  min-height: 620px;
}

.panel-title {
  font-size: 12px;
  font-weight: 600;
  color: rgba(255,255,255,0.42);
  letter-spacing: 0.5px;
}

.panel-empty {
  color: rgba(255,255,255,0.28);
  font-size: 13px;
  text-align: center;
  padding: 36px 16px;
}

.session-card {
  width: 100%;
  text-align: left;
  border: 1px solid rgba(255,255,255,0.05);
  background: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 12px;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

.session-card:hover {
  background: rgba(255,255,255,0.04);
  border-color: rgba(126,179,255,0.22);
  transform: translateY(-1px);
}

.session-card.active {
  background: rgba(79,142,247,0.12);
  border-color: rgba(79,142,247,0.28);
}

.session-row,
.session-meta,
.detail-subtitle,
.message-meta-line {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.session-title {
  flex: 1;
  font-size: 13px;
  font-weight: 600;
  color: #f3f7ff;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.session-time,
.session-kb,
.session-meta,
.detail-subtitle,
.message-meta-line {
  font-size: 11px;
  color: rgba(255,255,255,0.34);
}

.session-kb {
  margin-top: 6px;
}

.session-preview {
  margin-top: 8px;
  font-size: 12px;
  color: rgba(255,255,255,0.58);
  line-height: 1.5;
  min-height: 36px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.session-meta {
  margin-top: 10px;
}

.detail-header {
  padding: 18px 20px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.detail-title {
  font-size: 18px;
  font-weight: 700;
  color: #f4f7ff;
  word-break: break-word;
}

.detail-subtitle {
  margin-top: 8px;
  justify-content: flex-start;
  flex-wrap: wrap;
}

.detail-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.detail-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
}

.message-list {
  padding: 18px 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.message-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.message-row.user {
  flex-direction: row-reverse;
}

.message-badge {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  background: rgba(255,255,255,0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255,255,255,0.78);
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}

.message-body {
  max-width: min(720px, 100%);
  min-width: 0;
}

.message-row.user .message-body {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.message-bubble {
  border-radius: 16px;
  padding: 12px 14px;
  line-height: 1.7;
  font-size: 13px;
  color: rgba(255,255,255,0.88);
  border: 1px solid rgba(255,255,255,0.05);
}

.message-bubble.assistant {
  background: rgba(255,255,255,0.035);
}

.message-bubble.user {
  background: rgba(79,142,247,0.16);
}

.plain-text {
  white-space: pre-wrap;
}

.message-meta-line {
  margin-top: 6px;
  justify-content: flex-start;
  flex-wrap: wrap;
}

.message-row.user .message-meta-line {
  justify-content: flex-end;
}

:deep(.message-bubble img) {
  max-width: min(360px, 100%);
  border-radius: 10px;
}

.query-image {
  max-width: min(360px, 100%);
  border-radius: 10px;
  margin-bottom: 8px;
  display: block;
}

@media (max-width: 980px) {
  .stats-grid,
  .history-layout {
    grid-template-columns: 1fr;
  }

  .toolbar {
    flex-wrap: wrap;
  }

  .session-panel {
    max-height: none;
  }

  .detail-header {
    flex-direction: column;
  }
}

@media (max-width: 640px) {
  .stats-grid {
    gap: 10px;
  }

  .stat-card {
    padding: 16px;
  }

  .toolbar {
    align-items: stretch;
  }

  .toolbar :deep(.el-select),
  .toolbar :deep(.el-input),
  .toolbar :deep(.el-button) {
    width: 100% !important;
  }

  .history-layout {
    gap: 12px;
    min-height: 0;
  }

  .detail-panel {
    min-height: 460px;
  }

  .detail-header {
    padding: 16px;
  }

  .detail-actions {
    width: 100%;
    flex-wrap: wrap;
  }

  .detail-actions :deep(.el-button) {
    flex: 1 1 140px;
  }

  .message-list {
    padding: 14px 12px;
  }

  .message-row {
    gap: 8px;
  }

  .message-badge {
    width: 30px;
    height: 30px;
    border-radius: 10px;
  }
}
</style>
