<template>
  <div class="unanswered-page">
    <div class="stats-cards">
      <div class="stat-card">
        <div class="stat-value">{{ stats.total }}</div>
        <div class="stat-label">总回答数</div>
      </div>
      <div class="stat-card highlight">
        <div class="stat-value">{{ stats.unanswered }}</div>
        <div class="stat-label">未回答数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.unanswered_rate }}%</div>
        <div class="stat-label">未回答率</div>
      </div>
    </div>

    <div class="filter-bar">
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        size="small"
        style="width: 260px"
        value-format="YYYY-MM-DD"
      />
      <el-button type="primary" size="small" @click="loadData" :loading="loading">
        查询
      </el-button>
    </div>

    <el-table :data="items" v-loading="loading" style="width: 100%" stripe>
      <el-table-column type="index" width="50" />
      <el-table-column label="用户问题" min-width="200" show-overflow-tooltip>
        <template #default="{ row }">
          <div class="query-text">{{ row.query }}</div>
        </template>
      </el-table-column>
      <el-table-column label="AI 回复" min-width="180" show-overflow-tooltip>
        <template #default="{ row }">
          <span :class="{ 'fallback-text': row.used_fallback }">{{ row.answer }}</span>
        </template>
      </el-table-column>
      <el-table-column label="未回答原因" min-width="180" show-overflow-tooltip>
        <template #default="{ row }">
          <el-tag v-if="row.fallback_reason" type="danger" size="small">
            {{ row.fallback_reason }}
          </el-tag>
          <span v-else class="text-muted">—</span>
        </template>
      </el-table-column>
      <el-table-column label="质量等级" width="100">
        <template #default="{ row }">
          <el-tag
            :type="row.quality_level === 'HIGH' ? 'success' : row.quality_level === 'MEDIUM' ? 'warning' : 'danger'"
            size="small"
          >
            {{ row.quality_level || 'LOW' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="置信度" width="90">
        <template #default="{ row }">
          <span v-if="row.confidence != null">{{ (row.confidence * 100).toFixed(1) }}%</span>
          <span v-else class="text-muted">—</span>
        </template>
      </el-table-column>
      <el-table-column label="时间" width="160">
        <template #default="{ row }">
          {{ formatDate(row.created_at) }}
        </template>
      </el-table-column>
    </el-table>

    <div class="pagination-wrap">
      <el-pagination
        v-model:current-page="page"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next"
        @size-change="loadData"
        @current-change="loadData"
      />
    </div>

    <div v-if="dailyChartData.length > 0" class="chart-section">
      <h3>近 30 天趋势</h3>
      <div class="daily-list">
        <div v-for="d in dailyChartData" :key="d.date" class="daily-row">
          <span class="daily-date">{{ d.date }}</span>
          <div class="daily-bar-wrap">
            <div
              class="daily-bar"
              :style="{ width: dailyBarWidth(d.total) + '%' }"
            >
              <div
                class="daily-bar-inner"
                :style="{ width: dailyBarInnerWidth(d.unanswered, d.total) + '%' }"
              />
            </div>
          </div>
          <span class="daily-count">
            <span class="unanswered-count">{{ d.unanswered }}</span>
            / {{ d.total }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { apiService } from '../../services/api'

const loading = ref(false)
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const dateRange = ref([])
const stats = ref({ total: 0, unanswered: 0, unanswered_rate: 0 })
const dailyChartData = ref([])

const loadData = async () => {
  loading.value = true
  try {
    const params = {
      limit: pageSize.value,
      offset: (page.value - 1) * pageSize.value,
    }
    if (dateRange.value && dateRange.value.length === 2) {
      params.start_date = dateRange.value[0]
      params.end_date = dateRange.value[1]
    }

    const [msgRes, statRes] = await Promise.all([
      apiService.getUnansweredMessages(params),
      apiService.getConversationStats({
        start_date: params.start_date,
        end_date: params.end_date,
      }),
    ])

    if (msgRes.success) {
      items.value = msgRes.data.items
      total.value = msgRes.data.total
    }
    if (statRes.success) {
      const d = statRes.data
      const rate = d.unanswered_rate != null
        ? d.unanswered_rate
        : (d.total > 0 ? Math.round((d.unanswered / d.total) * 100 * 10) / 10 : 0)
      stats.value = { ...d, unanswered_rate: rate }
      dailyChartData.value = d.daily || []
    }
  } catch (e) {
    ElMessage.error('加载数据失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loading.value = false
  }
}

const formatDate = (s) => {
  if (!s) return '—'
  const d = new Date(s)
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

const maxDailyTotal = ref(1)
const dailyBarWidth = (total) => {
  if (!maxDailyTotal.value) return 0
  return Math.max((total / maxDailyTotal.value) * 100, 2)
}
const dailyBarInnerWidth = (unanswered, total) => {
  if (!total) return 0
  return (unanswered / total) * 100
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.unanswered-page {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
}
.stats-cards {
  display: flex; gap: 16px; margin-bottom: 24px;
}
.stat-card {
  flex: 1;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 20px 24px;
  text-align: center;
}
.stat-card.highlight {
  background: rgba(239,68,68,0.08);
  border-color: rgba(239,68,68,0.2);
}
.stat-value {
  font-size: 28px; font-weight: 700; color: #f0f4ff;
}
.stat-card.highlight .stat-value {
  color: #f87171;
}
.stat-label {
  font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 4px;
}
.filter-bar {
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 16px;
}
.fallback-text {
  color: #f87171;
}
.text-muted {
  color: rgba(255,255,255,0.25);
}
.pagination-wrap {
  display: flex; justify-content: flex-end; margin-top: 16px;
}
.chart-section {
  margin-top: 32px;
}
.chart-section h3 {
  font-size: 14px; color: rgba(255,255,255,0.5); margin-bottom: 12px;
}
.daily-list {
  display: flex; flex-direction: column; gap: 6px;
}
.daily-row {
  display: flex; align-items: center; gap: 10px;
  font-size: 12px;
}
.daily-date {
  width: 80px; color: rgba(255,255,255,0.35); text-align: right;
}
.daily-bar-wrap {
  flex: 1;
}
.daily-bar {
  height: 8px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden;
}
.daily-bar-inner {
  height: 100%; background: linear-gradient(90deg, #f87171, #ef4444); border-radius: 99px;
}
.daily-count {
  width: 80px; text-align: right; color: rgba(255,255,255,0.4);
}
.unanswered-count {
  color: #f87171; font-weight: 600;
}

@media (max-width: 768px) {
  .stats-cards {
    flex-wrap: wrap;
    gap: 10px;
  }

  .stat-card {
    flex: 1 1 160px;
    padding: 16px;
  }

  .filter-bar {
    align-items: stretch;
  }

  .filter-bar :deep(.el-select),
  .filter-bar :deep(.el-button) {
    width: 100% !important;
  }

  .daily-row {
    gap: 8px;
  }

  .daily-date,
  .daily-count {
    width: 64px;
  }
}
</style>
