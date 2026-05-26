<template>
  <el-card shadow="hover">
    <template #header>
      <div class="card-header">
        <span>知识库命中测试</span>
        <el-tag type="success" size="small">QueryContent</el-tag>
      </div>
    </template>

    <el-form :model="form" label-width="110px" style="margin-bottom: 8px">
      <el-row :gutter="16">
        <el-col :span="24">
          <el-form-item label="查询内容" required>
            <el-input
              v-model="form.query"
              placeholder="输入检索关键词或问题"
              clearable
              @keyup.enter="search"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="16">
        <el-col :span="6">
          <el-form-item label="TopK">
            <el-input-number v-model="form.topK" :min="1" :max="200" style="width: 100%" />
          </el-form-item>
        </el-col>
        <el-col :span="6">
          <el-form-item label="混合检索">
            <el-select v-model="form.hybridSearch" style="width: 100%">
              <el-option label="Weight" value="Weight" />
              <el-option label="RRF" value="RRF" />
              <el-option label="Cascaded" value="Cascaded" />
              <el-option label="关闭" value="" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col v-if="form.hybridSearch === 'Weight'" :span="6">
          <el-form-item label="向量权重">
            <el-input-number
              v-model="form.hybridAlpha"
              :min="0"
              :max="1"
              :step="0.1"
              :precision="1"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="6">
          <el-form-item label="Rerank">
            <el-switch v-model="form.rerank" active-text="开启" inactive-text="关闭" />
          </el-form-item>
        </el-col>
        <el-col v-if="form.rerank" :span="6">
          <el-form-item label="Rerank Top-N">
            <el-input-number v-model="form.rerankTopN" :min="1" :max="200" style="width: 100%" />
            <div class="minor-tip">留空时返回全部重排结果</div>
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="关键词预过滤">
            <el-input
              v-model="form.keywordFilter"
              placeholder="多个词用空格分隔，可选"
              clearable
            />
            <div class="minor-tip">会先做关键词过滤，再执行向量/全文检索。</div>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="标量过滤">
            <el-input
              v-model="form.filter"
              placeholder="SQL WHERE 格式，例如: file_name == 'guide.docx'"
              clearable
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row>
        <el-col :span="24" style="text-align: right">
          <el-button @click="reset">重置</el-button>
          <el-button type="primary" @click="search" :loading="searching" :disabled="!form.query">
            开始检索
          </el-button>
        </el-col>
      </el-row>
    </el-form>

    <template v-if="results.length > 0">
      <el-divider>
        <el-tag type="success">命中 {{ results.length }} 条</el-tag>
      </el-divider>

      <el-table :data="results" style="width: 100%" max-height="560" stripe border>
        <el-table-column type="expand">
          <template #default="{ row }">
            <div style="padding: 16px 24px">
              <div style="font-weight: 600; margin-bottom: 6px">完整内容</div>
              <div class="search-markdown" v-html="renderContent(row.content, row.image_map)" />
              <div v-if="row.metadata" style="margin-top: 8px; font-size: 12px; color: #606266">
                <span style="font-weight: 600">元数据：</span>{{ JSON.stringify(row.metadata) }}
              </div>
              <div v-if="row.loader_metadata" style="margin-top: 4px; font-size: 12px; color: #909399">
                <span style="font-weight: 600">加载元数据：</span>{{ row.loader_metadata }}
              </div>
              <div v-if="row.file_url" style="margin-top: 4px">
                <a :href="row.file_url" target="_blank" rel="noopener noreferrer" style="font-size: 12px">
                  文件链接
                </a>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column prop="file_name" label="文件名" width="180" show-overflow-tooltip />
        <el-table-column label="内容预览" min-width="280" show-overflow-tooltip>
          <template #default="{ row }">
            <span style="font-size: 13px">
              {{ previewText(row.content) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="来源" width="90" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="sourceType(row.retrieval_source)">
              {{ sourceLabel(row.retrieval_source) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="相似度" width="100" align="center">
          <template #default="{ row }">
            <span style="font-size: 13px; color: #409eff">
              {{ row.score != null ? row.score.toFixed(4) : '-' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="Rerank分数" width="110" align="center">
          <template #default="{ row }">
            <span
              v-if="row.rerank_score != null"
              style="font-size: 13px; color: #67c23a; font-weight: 600"
            >
              {{ row.rerank_score.toFixed(4) }}
            </span>
            <span v-else style="color: #c0c4cc">-</span>
          </template>
        </el-table-column>
      </el-table>
    </template>

    <el-empty v-else-if="!searching && searched" description="未命中任何结果" />
  </el-card>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import MarkdownIt from 'markdown-it'
import { docApi } from '@/services/docApi'

const props = defineProps({
  collection: { type: String, default: '' },
  retrievalConfig: { type: Object, default: () => ({}) },
})

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
  breaks: true,
})

const defaultLinkOpen =
  markdown.renderer.rules.link_open ||
  ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options))

markdown.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  tokens[idx].attrSet('target', '_blank')
  tokens[idx].attrSet('rel', 'noopener noreferrer')
  return defaultLinkOpen(tokens, idx, options, env, self)
}

const defaultForm = () => {
  const retrievalConfig = props.retrievalConfig || {}
  return {
    query: '',
    topK: retrievalConfig.rerank_enabled
      ? retrievalConfig.multi_doc_top_k ?? 20
      : retrievalConfig.llm_context_top_k ?? 10,
    hybridSearch: retrievalConfig.ranker ?? 'RRF',
    hybridAlpha: retrievalConfig.hybrid_alpha ?? 0.5,
    rerank: retrievalConfig.rerank_enabled ?? false,
    rerankTopN: retrievalConfig.multi_doc_rerank_top_k ?? 10,
    keywordFilter: '',
    filter: '',
  }
}

const form = ref(defaultForm())
const results = ref([])
const searching = ref(false)
const searched = ref(false)

const search = async () => {
  if (!form.value.query.trim()) {
    return
  }

  searching.value = true
  searched.value = false
  results.value = []

  try {
    const formData = new FormData()
    formData.append('query', form.value.query)
    formData.append('collection', props.collection || '')
    formData.append('top_k', String(form.value.topK))
    formData.append('hybrid_search', form.value.hybridSearch || 'RRF')
    formData.append('hybrid_alpha', String(form.value.hybridAlpha))

    if (form.value.rerank) {
      formData.append('rerank', 'true')
      formData.append('rerank_top_n', String(form.value.rerankTopN))
    }
    if (form.value.keywordFilter) {
      formData.append('keyword_filter', form.value.keywordFilter)
    }
    if (form.value.filter) {
      formData.append('filter_expr', form.value.filter)
    }

    const res = await docApi.searchDocuments(formData)
    results.value = res.data.data.results || []
    searched.value = true
    ElMessage.success(`命中 ${results.value.length} 条结果`)
  } catch (error) {
    ElMessage.error(`检索失败: ${error.response?.data?.detail || error.message}`)
  } finally {
    searching.value = false
  }
}

const reset = () => {
  form.value = defaultForm()
  results.value = []
  searched.value = false
}

const sourceLabel = (source) => ({ 1: '向量', 2: '全文', 3: '双路' }[source] || '-')
const sourceType = (source) => ({ 1: 'primary', 2: 'warning', 3: 'success' }[source] || 'info')

const PLACEHOLDER_RE = /<<IMAGE:[0-9a-f]+>>/g

const stripPlaceholders = (content) => (content || '').replace(PLACEHOLDER_RE, '')

const previewText = (content) => {
  const text = stripPlaceholders(content).replace(/\s+/g, ' ').trim()
  return text.length > 120 ? `${text.slice(0, 120)}...` : text
}

const renderContent = (content, imageMap = {}) => {
  if (!content) {
    return ''
  }

  const markdownSource = content.replace(PLACEHOLDER_RE, (placeholder) => {
    const url = imageMap?.[placeholder]
    return url ? `\n\n![图片](<${url}>)\n\n` : '\n\n[图片缺失]\n\n'
  })

  return markdown.render(markdownSource)
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.minor-tip {
  margin-top: 4px;
  font-size: 11px;
  color: #909399;
}

.search-markdown {
  background: #0d1117;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 12px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.7;
  color: #cbd5e1;
  overflow-x: auto;
  word-break: break-word;
}

.search-markdown :deep(p) {
  margin: 6px 0;
}

.search-markdown :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
}

.search-markdown :deep(th),
.search-markdown :deep(td) {
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 6px 10px;
  vertical-align: top;
}

.search-markdown :deep(th) {
  background: rgba(255, 255, 255, 0.05);
  font-weight: 600;
}

.search-markdown :deep(a) {
  color: #7eb3ff;
  text-decoration: none;
}

.search-markdown :deep(a:hover) {
  text-decoration: underline;
}

.search-markdown :deep(img) {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 10px 0;
  border-radius: 8px;
}

.search-markdown :deep(code) {
  background: rgba(79, 142, 247, 0.12);
  border-radius: 4px;
  padding: 1px 5px;
}

.search-markdown :deep(pre) {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 12px;
  overflow-x: auto;
}

.search-markdown :deep(pre code) {
  background: transparent;
  padding: 0;
}
</style>
