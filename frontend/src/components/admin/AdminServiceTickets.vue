<template>
  <div class="service-ticket-page">
    <div class="ticket-stats">
      <button
        v-for="card in statCards"
        :key="card.key"
        class="ticket-stat"
        :class="{ active: filters.status === card.status, warn: card.key === 'pending_manual', ok: card.key === 'resolved_ai' }"
        @click="selectStatus(card.status)"
      >
        <span class="stat-value">{{ card.value }}</span>
        <span class="stat-label">{{ card.label }}</span>
      </button>
    </div>

    <div class="ticket-filters">
      <el-select v-model="filters.status" placeholder="状态" size="small" clearable style="width: 150px" @change="reloadFirstPage">
        <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
      </el-select>
      <el-input
        v-model="filters.kb_name"
        placeholder="知识库"
        size="small"
        clearable
        style="width: 160px"
        @keyup.enter="reloadFirstPage"
        @clear="reloadFirstPage"
      />
      <el-input
        v-model="filters.user_id"
        placeholder="用户 ID"
        size="small"
        clearable
        style="width: 180px"
        @keyup.enter="reloadFirstPage"
        @clear="reloadFirstPage"
      />
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        size="small"
        value-format="YYYY-MM-DD"
        style="width: 260px"
        @change="reloadFirstPage"
      />
      <el-button type="primary" size="small" :loading="loading" @click="reloadFirstPage">
        <el-icon><Search /></el-icon>
        查询
      </el-button>
      <el-button size="small" :loading="loading" @click="loadData">
        <el-icon><Refresh /></el-icon>
        刷新
      </el-button>
      <el-button
        type="danger"
        size="small"
        plain
        :disabled="!selectedCount || batchDeleting"
        :loading="batchDeleting"
        @click="confirmDeleteSelectedTickets"
      >
        <el-icon><Delete /></el-icon>
        批量删除<span v-if="selectedCount">（{{ selectedCount }}）</span>
      </el-button>
    </div>

    <el-table
      ref="tableRef"
      :data="items"
      v-loading="loading"
      stripe
      class="ticket-table"
      @row-click="handleRowClick"
      @selection-change="handleSelectionChange"
    >
      <el-table-column type="selection" width="52" align="center" />
      <el-table-column label="状态" width="110">
        <template #default="{ row }">
          <el-tag :type="statusMeta(row.status).type" size="small">{{ statusMeta(row.status).label }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="店长 / 用户" width="160" show-overflow-tooltip>
        <template #default="{ row }">
          <div class="user-cell">{{ row.user_name || row.user_id || 'guest' }}</div>
          <div v-if="row.entry_user_name || row.entry_user_id" class="muted mini">
            {{ row.entry_user_name || row.entry_user_id }}
          </div>
          <div class="muted mini">{{ row.channel || 'web' }}<template v-if="row.entry_source"> · {{ row.entry_source }}</template></div>
        </template>
      </el-table-column>
      <el-table-column label="问题" min-width="230" show-overflow-tooltip>
        <template #default="{ row }">
          <span class="query-text">{{ row.query }}</span>
        </template>
      </el-table-column>
      <el-table-column label="AI 答案" min-width="260" show-overflow-tooltip>
        <template #default="{ row }">
          <span class="answer-text">{{ row.answer || '—' }}</span>
        </template>
      </el-table-column>
      <el-table-column label="知识库" width="130" show-overflow-tooltip>
        <template #default="{ row }">{{ row.kb_name || '—' }}</template>
      </el-table-column>
      <el-table-column label="置信度" width="90">
        <template #default="{ row }">
          <span v-if="row.confidence !== null && row.confidence !== undefined">{{ Math.round(row.confidence * 100) }}%</span>
          <span v-else class="muted">—</span>
        </template>
      </el-table-column>
      <el-table-column label="召回" width="80">
        <template #default="{ row }">{{ row.context_count || 0 }}</template>
      </el-table-column>
      <el-table-column label="时间" width="150">
        <template #default="{ row }">{{ formatDate(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="140" fixed="right">
        <template #default="{ row }">
          <el-button size="small" link type="primary" @click.stop="openDetail(row)">详情</el-button>
          <el-button
            size="small"
            link
            type="danger"
            :loading="deletingTicketId === row.id"
            @click.stop="confirmDeleteTicket(row)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="pagination-wrap">
      <el-pagination
        v-model:current-page="page"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next"
        @size-change="loadData"
        @current-change="loadData"
      />
    </div>

    <el-drawer v-model="detailVisible" size="min(760px, 92vw)" :with-header="false" destroy-on-close>
      <div v-if="detail" class="ticket-detail">
        <div class="detail-head">
          <div>
            <div class="detail-title">服务记录详情</div>
            <div class="muted">{{ formatDate(detail.created_at) }} · {{ detail.kb_name || '未绑定知识库' }}</div>
          </div>
          <div class="detail-head-actions">
            <el-tag :type="statusMeta(detail.status).type">{{ statusMeta(detail.status).label }}</el-tag>
            <el-button
              type="danger"
              plain
              size="small"
              :loading="deletingTicketId === detail.id"
              @click="confirmDeleteTicket(detail)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
            <el-button circle size="small" @click="detailVisible = false">
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
        </div>

        <section class="detail-section">
          <div class="section-title">问题</div>
          <div class="text-block">{{ detail.query }}</div>
        </section>

        <section class="detail-section">
          <div class="section-title">入口身份</div>
          <div class="clarification-grid">
            <div>
              <span class="muted mini">工单用户</span>
              <strong>{{ detail.user_name || detail.user_id || 'guest' }}</strong>
            </div>
            <div>
              <span class="muted mini">入口用户 ID</span>
              <strong>{{ detail.entry_user_id || '—' }}</strong>
            </div>
            <div>
              <span class="muted mini">入口用户名称</span>
              <strong>{{ detail.entry_user_name || '—' }}</strong>
            </div>
            <div>
              <span class="muted mini">入口来源</span>
              <strong>{{ detail.entry_source || detail.channel || 'web' }}</strong>
            </div>
          </div>
        </section>

        <section v-if="chatHistory.length" class="detail-section">
          <div class="section-title">AI 与用户对话记录</div>
          <div class="chat-history-list">
            <div
              v-for="(message, index) in chatHistory"
              :key="message.id || `${message.role}-${index}`"
              class="chat-history-item"
              :class="message.role === 'user' ? 'from-user' : 'from-ai'"
            >
              <div class="chat-history-meta">
                <span class="chat-role">{{ chatRoleLabel(message.role) }}</span>
                <span v-if="message.created_at" class="muted mini">{{ formatDate(message.created_at) }}</span>
                <button
                  v-if="message.query_image_oss_key"
                  type="button"
                  class="image-badge image-badge-button"
                  :disabled="!queryImageUrl(message.query_image_oss_key)"
                  @click.stop="openImagePreview(queryImageUrl(message.query_image_oss_key), '用户截图')"
                >
                  图片
                </button>
              </div>
              <div class="chat-history-content">{{ message.content || '（空消息）' }}</div>
              <div v-if="messageImagePlaceholders(message).length" class="image-strip">
                <button
                  v-for="ph in messageImagePlaceholders(message)"
                  :key="ph"
                  type="button"
                  class="image-thumb-button"
                  :disabled="!imageUrl(ph)"
                  @click.stop="openImagePreview(imageUrl(ph), ph)"
                >
                  <img v-if="imageUrl(ph)" :src="imageUrl(ph)" :alt="ph" />
                  <span v-else>{{ ph }}</span>
                </button>
              </div>
              <div v-if="message.sources?.length" class="chat-source-list">
                <span
                  v-for="source in message.sources"
                  :key="`${source.file_name || source.title || source.chunk_id || 'source'}-${source.chunk_index ?? ''}`"
                >
                  {{ source.file_name || source.title || '召回来源' }}<template v-if="source.chunk_index !== undefined && source.chunk_index !== null"> #{{ source.chunk_index }}</template>
                </span>
              </div>
            </div>
          </div>
        </section>

        <section class="detail-section">
          <div class="section-title">AI 答案 / 人工处理</div>
          <el-form label-position="top">
            <el-form-item label="状态">
              <el-select v-model="editForm.status" style="width: 220px">
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item label="答案">
              <el-input v-model="editForm.answer" type="textarea" :rows="5" />
            </el-form-item>
            <el-form-item label="处理备注">
              <el-input v-model="editForm.note" type="textarea" :rows="3" placeholder="可记录人工判断、补充说明或回访结果" />
            </el-form-item>
            <div class="detail-actions">
              <el-button type="primary" :loading="ticketSaving" @click="saveTicket">
                <el-icon><Check /></el-icon>
                保存处理
              </el-button>
            </div>
          </el-form>
        </section>

        <section v-if="detail.clarification && Object.keys(detail.clarification).length" class="detail-section">
          <div class="section-title">澄清采集信息</div>
          <div class="clarification-grid">
            <div>
              <span class="muted mini">发送人 ID</span>
              <strong>{{ detail.sender_id || detail.clarification.collected?.sender_id || '未提供' }}</strong>
            </div>
            <div>
              <span class="muted mini">具体提问者</span>
              <strong>{{ detail.requester_name || detail.clarification.collected?.requester_name || detail.user_name || '未提供' }}</strong>
            </div>
            <div>
              <span class="muted mini">澄清轮次</span>
              <strong>{{ detail.clarification_round || detail.clarification.turns?.length || 0 }}</strong>
            </div>
            <div>
              <span class="muted mini">缺失信息</span>
              <strong>{{ fieldLabels(detail.clarification.missing_fields).join('、') || '无' }}</strong>
            </div>
          </div>
          <div class="collected-block">
            <div v-for="item in collectedRows" :key="item.label" class="collected-row">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
          </div>
          <div v-if="collectedImageKeys.length" class="image-strip">
            <button
              v-for="key in collectedImageKeys"
              :key="key"
              type="button"
              class="image-thumb-button"
              :disabled="!queryImageUrl(key)"
              @click.stop="openImagePreview(queryImageUrl(key), '截图')"
            >
              <img v-if="queryImageUrl(key)" :src="queryImageUrl(key)" alt="截图" />
              <span v-else>截图</span>
            </button>
          </div>
          <div v-if="detail.clarification.turns?.length" class="turn-list">
            <div v-for="turn in detail.clarification.turns" :key="turn.round" class="turn-item">
              <span class="turn-round">第 {{ turn.round }} 轮</span>
              <span class="turn-query">{{ turn.query }}</span>
            </div>
          </div>
        </section>

        <section class="detail-section">
          <div class="section-title with-action">
            <span>召回上下文快照</span>
            <el-button size="small" type="success" :disabled="!hasVectorizableContext" :loading="revectorizing" @click="revectorize">
              <el-icon><RefreshRight /></el-icon>
              重新向量化
            </el-button>
          </div>

          <el-empty v-if="!detail.contexts?.length" description="本记录没有召回上下文，可走人工答案入库兜底路径" />

          <div v-for="ctx in detail.contexts" :key="ctx.id" class="context-item">
            <div class="context-meta">
              <span>{{ ctx.file_name || '未知文件' }}</span>
              <span>#{{ ctx.chunk_index ?? '-' }}</span>
              <span v-if="ctx.metadata?.chunk_strategy === 'parent_child'">父子块</span>
              <span v-if="ctx.metadata?.parent_id">父块 {{ ctx.metadata.parent_id }}</span>
              <span v-if="ctx.score !== null && ctx.score !== undefined">分数 {{ Number(ctx.score).toFixed(3) }}</span>
              <span v-if="ctx.job_id">job {{ shortId(ctx.job_id) }}</span>
            </div>
            <div v-if="parentContent(ctx)" class="parent-context">
              <div class="context-subtitle">父块上下文</div>
              <div class="parent-context-body">{{ parentContent(ctx) }}</div>
            </div>
            <div v-if="contextImagePlaceholders(ctx).length" class="image-strip context-image-strip">
              <button
                v-for="ph in contextImagePlaceholders(ctx)"
                :key="`${ctx.id || ctx.chunk_id}-${ph}`"
                type="button"
                class="image-thumb-button"
                :disabled="!imageUrl(ph)"
                @click.stop="openImagePreview(imageUrl(ph), ph)"
              >
                <img v-if="imageUrl(ph)" :src="imageUrl(ph)" :alt="ph" />
                <span v-else>{{ ph }}</span>
              </button>
            </div>
            <div class="context-subtitle">子块内容</div>
            <el-input
              v-model="contextEdits[ctx.chunk_id]"
              type="textarea"
              :rows="7"
              :disabled="!ctx.chunk_id"
            />
            <div class="context-actions">
              <span class="muted mini">chunk {{ ctx.chunk_id ? shortId(ctx.chunk_id) : '—' }}</span>
              <el-button size="small" type="primary" plain :disabled="!ctx.chunk_id" :loading="savingChunkId === ctx.chunk_id" @click="saveChunk(ctx)">
                <el-icon><EditPen /></el-icon>
                保存切片
              </el-button>
            </div>
          </div>

          <div v-if="vectorizeResult" class="vectorize-result">
            <div v-if="vectorizeResult.succeeded?.length">成功：{{ vectorizeResult.succeeded.map(x => shortId(x.job_id)).join(', ') }}</div>
            <div v-if="vectorizeResult.failed?.length" class="danger">失败：{{ vectorizeResult.failed.map(x => `${shortId(x.job_id)} ${x.error}`).join('；') }}</div>
          </div>
        </section>
      </div>
    </el-drawer>

    <el-dialog
      v-model="imagePreview.visible"
      :title="imagePreview.title"
      width="min(920px, 92vw)"
      append-to-body
      class="service-ticket-image-dialog"
    >
      <img v-if="imagePreview.url" class="preview-image" :src="imagePreview.url" :alt="imagePreview.title" />
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Check, Close, Delete, EditPen, Refresh, RefreshRight, Search } from '@element-plus/icons-vue'
import { apiService } from '../../services/api'

const statusOptions = [
  { value: 'clarifying', label: '澄清中', type: 'warning' },
  { value: 'resolved_ai', label: 'AI 已解决', type: 'success' },
  { value: 'unanswered_normal', label: '常规未命中', type: 'info' },
  { value: 'pending_manual', label: '待人工', type: 'danger' },
  { value: 'in_progress', label: '处理中', type: 'warning' },
  { value: 'resolved_manual', label: '人工已解决', type: 'success' },
  { value: 'ignored', label: '已忽略', type: 'info' },
]

const statusMap = Object.fromEntries(statusOptions.map(item => [item.value, item]))
const statusMeta = (status) => statusMap[status] || { label: status || '未知', type: 'info' }

const loading = ref(false)
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const dateRange = ref([])
const filters = ref({ status: '', kb_name: '', user_id: '' })
const stats = ref({ total: 0, by_status: {}, daily: [] })

const detailVisible = ref(false)
const detail = ref(null)
const detailLoading = ref(false)
const ticketSaving = ref(false)
const deletingTicketId = ref('')
const batchDeleting = ref(false)
const savingChunkId = ref('')
const revectorizing = ref(false)
const vectorizeResult = ref(null)
const contextEdits = ref({})
const editForm = ref({ status: '', answer: '', note: '' })
const imageUrlMap = ref({})
const queryImageUrlMap = ref({})
const imagePreview = ref({ visible: false, url: '', title: '图片预览' })
const tableRef = ref(null)
const selectedTickets = ref([])

const statCards = computed(() => [
  { key: 'all', label: '全部', value: stats.value.total || 0, status: '' },
  { key: 'pending_manual', label: '待人工', value: stats.value.by_status?.pending_manual || 0, status: 'pending_manual' },
  { key: 'unanswered_normal', label: '常规未命中', value: stats.value.by_status?.unanswered_normal || 0, status: 'unanswered_normal' },
  { key: 'resolved_ai', label: 'AI 已解决', value: stats.value.by_status?.resolved_ai || 0, status: 'resolved_ai' },
  { key: 'resolved_manual', label: '人工已解决', value: stats.value.by_status?.resolved_manual || 0, status: 'resolved_manual' },
  { key: 'clarifying', label: '澄清中', value: stats.value.by_status?.clarifying || 0, status: 'clarifying' },
])
const selectedCount = computed(() => selectedTickets.value.length)

const hasVectorizableContext = computed(() =>
  Boolean(detail.value?.contexts?.some(ctx => ctx.job_id))
)

const fieldLabelMap = {
  issue_detail: '问题详情',
  phone: '手机号',
  id_card: '身份证/主体证件',
  image: '截图',
}
const fieldLabels = (fields = []) => fields.map(field => fieldLabelMap[field] || field)

const collectedRows = computed(() => {
  const c = detail.value?.clarification?.collected || {}
  return [
    { label: '问题详情', value: c.issue_details?.join('；') || c.issue_detail || '—' },
    { label: '手机号', value: c.phone || '—' },
    { label: '身份证/主体证件', value: c.id_card || '—' },
    { label: '截图', value: c.image_keys?.length ? `${c.image_keys.length} 张` : (c.has_image ? '已提供' : '—') },
  ]
})

const chatHistory = computed(() => detail.value?.clarification?.chat_history || [])

const collectedImageKeys = computed(() => {
  const clarification = detail.value?.clarification || {}
  const collected = clarification.collected || {}
  const keys = [
    ...(Array.isArray(collected.image_keys) ? collected.image_keys : []),
    collected.query_image_oss_key,
    clarification.image_analysis?.query_image_oss_key,
  ].filter(Boolean)
  return [...new Set(keys)]
})

const chatRoleLabel = (role) => {
  if (role === 'user') return '用户'
  if (role === 'assistant') return 'AI'
  return role || '消息'
}

const extractImagePlaceholders = (text = '') => {
  const matches = String(text || '').match(/<<IMAGE:[0-9a-f]+>>/gi) || []
  return [...new Set(matches)]
}

const imageUrl = (placeholder) => imageUrlMap.value[placeholder] || ''
const queryImageUrl = (key) => queryImageUrlMap.value[key] || ''

const messageImagePlaceholders = (message) => extractImagePlaceholders(message?.content || '')

const contextImagePlaceholders = (ctx) => [
  ...new Set([
    ...extractImagePlaceholders(parentContent(ctx)),
    ...extractImagePlaceholders(ctx?.content || ''),
  ]),
]

const openImagePreview = (url, title = '图片预览') => {
  if (!url) {
    ElMessage.warning('图片链接还未加载或已失效，请刷新后重试')
    return
  }
  imagePreview.value = { visible: true, url, title }
}

const resolveDetailImages = async (ticket) => {
  const placeholders = new Set()
  const ossKeys = new Set()
  const addPlaceholders = (text) => {
    extractImagePlaceholders(text).forEach(ph => placeholders.add(ph))
  }
  addPlaceholders(ticket?.query)
  addPlaceholders(ticket?.answer)
  for (const message of ticket?.clarification?.chat_history || []) {
    addPlaceholders(message?.content)
    if (message?.query_image_oss_key) ossKeys.add(message.query_image_oss_key)
  }
  for (const turn of ticket?.clarification?.turns || []) {
    addPlaceholders(turn?.query)
    addPlaceholders(turn?.answer)
    if (turn?.image_key) ossKeys.add(turn.image_key)
  }
  const collected = ticket?.clarification?.collected || {}
  for (const key of collected.image_keys || []) {
    if (key) ossKeys.add(key)
  }
  if (collected.query_image_oss_key) ossKeys.add(collected.query_image_oss_key)
  if (ticket?.clarification?.image_analysis?.query_image_oss_key) {
    ossKeys.add(ticket.clarification.image_analysis.query_image_oss_key)
  }
  for (const ctx of ticket?.contexts || []) {
    addPlaceholders(ctx?.content)
    addPlaceholders(ctx?.metadata?.parent_content)
    addPlaceholders(ctx?.metadata?.parentContent)
  }

  const [imageRes, queryImageRes] = await Promise.allSettled([
    placeholders.size ? apiService.resolveImages([...placeholders]) : Promise.resolve({ success: true, data: {} }),
    ossKeys.size ? apiService.resolveQueryImages([...ossKeys]) : Promise.resolve({ success: true, data: {} }),
  ])
  imageUrlMap.value = imageRes.status === 'fulfilled' ? (imageRes.value.data || {}) : {}
  queryImageUrlMap.value = queryImageRes.status === 'fulfilled' ? (queryImageRes.value.data || {}) : {}
}

const buildParams = () => {
  const params = {
    limit: pageSize.value,
    offset: (page.value - 1) * pageSize.value,
  }
  if (filters.value.status) params.status = filters.value.status
  if (filters.value.kb_name) params.kb_name = filters.value.kb_name.trim()
  if (filters.value.user_id) params.user_id = filters.value.user_id.trim()
  if (dateRange.value?.length === 2) {
    params.start_date = dateRange.value[0]
    params.end_date = dateRange.value[1]
  }
  return params
}

const loadStats = async () => {
  const params = buildParams()
  delete params.status
  delete params.limit
  delete params.offset
  const res = await apiService.getServiceTicketStats(params)
  if (res.success) stats.value = res.data
}

const loadData = async () => {
  loading.value = true
  try {
    const [listRes] = await Promise.all([
      apiService.getServiceTickets(buildParams()),
      loadStats(),
    ])
    if (listRes.success) {
      items.value = listRes.data.items || []
      total.value = listRes.data.total || 0
      await nextTick()
      tableRef.value?.clearSelection?.()
      selectedTickets.value = []
    }
  } catch (e) {
    ElMessage.error('加载服务记录失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loading.value = false
  }
}

const reloadFirstPage = () => {
  page.value = 1
  loadData()
}

const selectStatus = (status) => {
  filters.value.status = status
  reloadFirstPage()
}

const handleSelectionChange = (rows) => {
  selectedTickets.value = (rows || []).filter(ticket => ticket?.id)
}

const handleRowClick = (row, column) => {
  if (column?.type === 'selection') return
  openDetail(row)
}

const openDetail = async (row) => {
  detailVisible.value = true
  detailLoading.value = true
  vectorizeResult.value = null
  imageUrlMap.value = {}
  queryImageUrlMap.value = {}
  try {
    const res = await apiService.getServiceTicket(row.id)
    if (res.success) {
      detail.value = res.data
      await resolveDetailImages(res.data)
      editForm.value = {
        status: res.data.status || 'pending_manual',
        answer: res.data.answer || '',
        note: res.data.note || '',
      }
      contextEdits.value = Object.fromEntries(
        (res.data.contexts || [])
          .filter(ctx => ctx.chunk_id)
          .map(ctx => [ctx.chunk_id, ctx.content || '']),
      )
    }
  } catch (e) {
    ElMessage.error('加载详情失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    detailLoading.value = false
  }
}

const saveTicket = async () => {
  if (!detail.value) return
  ticketSaving.value = true
  try {
    const res = await apiService.updateServiceTicket(detail.value.id, editForm.value)
    if (res.success) {
      detail.value = { ...detail.value, ...res.data }
      ElMessage.success('服务记录已保存')
      await loadData()
    }
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    ticketSaving.value = false
  }
}

const confirmDeleteTicket = async (ticket) => {
  if (!ticket?.id) return
  const label = ticket.query ? `「${String(ticket.query).slice(0, 28)}」` : shortId(ticket.id)
  try {
    await ElMessageBox.confirm(
      `确定删除工单 ${label} 吗？删除后该工单及召回上下文记录将不可恢复。`,
      '删除工单',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger',
      },
    )
  } catch {
    return
  }

  deletingTicketId.value = ticket.id
  try {
    const res = await apiService.deleteServiceTicket(ticket.id)
    if (res.success) {
      ElMessage.success('工单已删除')
      if (detail.value?.id === ticket.id) {
        detailVisible.value = false
        detail.value = null
      }
      if (items.value.length === 1 && page.value > 1) {
        page.value -= 1
      }
      await loadData()
    }
  } catch (e) {
    ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    deletingTicketId.value = ''
  }
}

const confirmDeleteSelectedTickets = async () => {
  if (!selectedTickets.value.length || batchDeleting.value) return

  const ticketsToDelete = [...selectedTickets.value]
  const count = ticketsToDelete.length

  try {
    await ElMessageBox.confirm(
      `确定删除选中的 ${count} 条工单吗？删除后相关工单及召回上下文记录将不可恢复。`,
      '批量删除工单',
      {
        confirmButtonText: '批量删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger',
      },
    )
  } catch {
    return
  }

  batchDeleting.value = true

  const failedTickets = []
  let successCount = 0

  try {
    for (const ticket of ticketsToDelete) {
      try {
        const res = await apiService.deleteServiceTicket(ticket.id)
        if (res.success) {
          successCount += 1
        } else {
          failedTickets.push(ticket)
        }
      } catch {
        failedTickets.push(ticket)
      }
    }

    if (successCount > 0) {
      ElMessage.success(`已删除 ${successCount} 条工单`)
    }

    if (failedTickets.length > 0) {
      ElMessage.error(`有 ${failedTickets.length} 条工单删除失败，请稍后重试`)
    }

    if (detail.value && ticketsToDelete.some(ticket => ticket.id === detail.value.id)) {
      detailVisible.value = false
      detail.value = null
    }

    if (successCount >= items.value.length && page.value > 1) {
      page.value -= 1
    }

    await loadData()
  } finally {
    batchDeleting.value = false
  }
}

const saveChunk = async (ctx) => {
  if (!detail.value || !ctx.chunk_id) return
  const content = contextEdits.value[ctx.chunk_id] || ''
  if (content.trim().length < 2) {
    ElMessage.warning('切片内容不能为空')
    return
  }
  savingChunkId.value = ctx.chunk_id
  try {
    const res = await apiService.updateServiceTicketChunk(detail.value.id, ctx.chunk_id, content)
    if (res.success) {
      ElMessage.success('切片已保存，所属任务已标记为待重新向量化')
      ctx.content = content
    }
  } catch (e) {
    ElMessage.error('保存切片失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    savingChunkId.value = ''
  }
}

const parentContent = (ctx) => {
  const metadata = ctx?.metadata || {}
  return metadata.parent_content || metadata.parentContent || ''
}

const revectorize = async () => {
  if (!detail.value) return
  revectorizing.value = true
  try {
    const res = await apiService.revectorizeServiceTicket(detail.value.id)
    if (res.success) {
      vectorizeResult.value = res.data
      const failed = res.data.failed?.length || 0
      if (failed) ElMessage.warning('重新向量化完成，存在失败任务')
      else ElMessage.success('相关切片已重新写入 Milvus')
    }
  } catch (e) {
    ElMessage.error('重新向量化失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    revectorizing.value = false
  }
}

const shortId = (id) => String(id || '').slice(0, 8)

const formatDate = (s) => {
  if (!s) return '—'
  return new Date(s).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

onMounted(loadData)
</script>

<style scoped>
.service-ticket-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}

.ticket-stats {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
}

.ticket-stat {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  min-height: 74px;
  padding: 14px;
  border-radius: 8px;
  border: 1px solid #dce5ee;
  background: rgba(255,255,255,0.94);
  color: #213041;
  box-shadow: 0 16px 36px rgba(31, 45, 61, 0.06);
  cursor: pointer;
  text-align: left;
}

.ticket-stat:hover,
.ticket-stat.active {
  border-color: rgba(79,142,247,0.34);
  background: #f3f8ff;
}

.ticket-stat.warn .stat-value { color: #d69a1d; }
.ticket-stat.ok .stat-value { color: #22b07d; }
.stat-value { font-size: 24px; font-weight: 800; line-height: 1; }
.stat-label { font-size: 12px; color: #98a6b3; }

.ticket-filters {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.ticket-table {
  --el-table-border-color: #e1e8ef;
  border-radius: 8px;
  overflow: hidden;
}

.user-cell,
.query-text,
.answer-text {
  color: #213041;
}

.muted {
  color: #98a6b3;
}

.mini {
  margin-top: 2px;
  font-size: 11px;
}

.pagination-wrap {
  display: flex;
  justify-content: flex-end;
}

.ticket-detail {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 22px;
}

.detail-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: flex-start;
}

.detail-head-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.detail-title {
  color: #526679;
  font-size: 18px;
  font-weight: 800;
}

.detail-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 14px;
  border-top: 1px solid #e8eef4;
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  color: #526679;
  font-size: 14px;
  font-weight: 700;
}

.section-title.with-action {
  flex-wrap: wrap;
}

.text-block {
  white-space: pre-wrap;
  padding: 12px;
  border-radius: 8px;
  background: #f7fafc;
  border: 1px solid #e2eaf2;
  color: #213041;
  line-height: 1.7;
}

.chat-history-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chat-history-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border: 1px solid #e2eaf2;
  border-left-width: 2px;
  border-radius: 8px;
  background: rgba(255,255,255,0.9);
}

.chat-history-item.from-user {
  border-left-color: rgba(126,179,255,0.62);
}

.chat-history-item.from-ai {
  border-left-color: rgba(45,212,160,0.48);
}

.chat-history-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.chat-role {
  color: #526679;
  font-size: 12px;
  font-weight: 800;
}

.image-badge {
  padding: 2px 6px;
  border-radius: 999px;
  border: 0;
  background: #eaf3ff;
  color: #4f8ef7;
  font-size: 11px;
}

.image-badge-button {
  cursor: pointer;
}

.image-badge-button:disabled {
  cursor: not-allowed;
  opacity: 0.52;
}

.image-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: flex-start;
}

.context-image-strip {
  margin-top: 2px;
}

.image-thumb-button {
  width: 96px;
  height: 72px;
  padding: 0;
  overflow: hidden;
  border: 1px solid rgba(79,142,247,0.2);
  border-radius: 8px;
  background: #f9fbfd;
  color: #98a6b3;
  cursor: pointer;
}

.image-thumb-button:hover:not(:disabled) {
  border-color: rgba(126,179,255,0.72);
}

.image-thumb-button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.image-thumb-button img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}

.image-thumb-button span {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 6px;
  font-size: 11px;
  line-height: 1.35;
  word-break: break-all;
}

.preview-image {
  display: block;
  max-width: 100%;
  max-height: 78vh;
  margin: 0 auto;
  border-radius: 8px;
  object-fit: contain;
}

.chat-history-content {
  white-space: pre-wrap;
  word-break: break-word;
  color: #213041;
  font-size: 13px;
  line-height: 1.65;
}

.chat-source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chat-source-list span {
  padding: 2px 6px;
  border-radius: 6px;
  background: #eff4f8;
  color: #5d6f80;
  font-size: 11px;
}

.detail-actions,
.context-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
}

.context-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border: 1px solid #e2eaf2;
  border-radius: 8px;
  background: rgba(255,255,255,0.92);
}

.context-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: #98a6b3;
  font-size: 12px;
}

.context-subtitle {
  color: #617486;
  font-size: 12px;
  font-weight: 700;
}

.parent-context {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-left: 10px;
  border-left: 2px solid rgba(34,176,125,0.28);
}

.parent-context-body {
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  color: #7a8b9b;
  font-size: 12px;
  line-height: 1.7;
}

.vectorize-result {
  padding: 10px 12px;
  border-radius: 8px;
  background: #eef9f4;
  border: 1px solid rgba(34,176,125,0.18);
  color: #1f8c65;
  font-size: 12px;
  line-height: 1.7;
}

.clarification-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.clarification-grid > div,
.collected-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border: 1px solid #e2eaf2;
  border-radius: 8px;
  background: #fbfcfe;
}

.clarification-grid strong,
.collected-row strong {
  color: #213041;
  font-size: 13px;
  word-break: break-word;
}

.collected-block {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.turn-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.turn-item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 8px 10px;
  border-radius: 8px;
  background: #f7fafc;
  color: #617486;
}

.turn-round {
  flex: 0 0 auto;
  color: #1f8c65;
  font-size: 12px;
  font-weight: 700;
}

.turn-query {
  min-width: 0;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
}

.danger {
  color: #fca5a5;
}

@media (max-width: 900px) {
  .ticket-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .ticket-filters :deep(.el-input),
  .ticket-filters :deep(.el-select),
  .ticket-filters :deep(.el-date-editor) {
    width: 100% !important;
  }

  .clarification-grid,
  .collected-block {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 520px) {
  .ticket-stats {
    grid-template-columns: 1fr;
  }

  .pagination-wrap {
    justify-content: flex-start;
    overflow-x: auto;
  }
}
</style>
