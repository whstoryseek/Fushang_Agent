<template>
  <div class="admin-data-view">
    <div class="view-toolbar">
      <el-select
        v-model="selectedCollection"
        placeholder="选择知识库"
        style="width:260px"
        :loading="loading"
      >
        <el-option
          v-for="col in collections"
          :key="col.name"
          :label="col.display_name || col.name"
          :value="col.name"
        />
      </el-select>
      <el-button type="primary" plain :loading="loading" @click="loadCollections">刷新</el-button>
    </div>

    <el-empty v-if="!selectedCollection" description="请选择知识库" />
    <el-tabs v-else type="border-card">
      <el-tab-pane label="文件列表" name="files">
        <DocList
          ref="docListRef"
          :collection="selectedCollection"
          @view-chunks="openChunkViewer"
        />
      </el-tab-pane>
      <el-tab-pane label="切片检索" name="search">
        <DocSearch
          :collection="selectedCollection"
          :retrieval-config="selectedKb?.retrieval_config || {}"
        />
      </el-tab-pane>
    </el-tabs>

    <el-dialog
      v-model="chunkDialogVisible"
      title="查看切片"
      width="92vw"
      top="4vh"
      destroy-on-close
      class="chunk-view-dialog"
    >
      <ChunkEditorPanel
        v-if="selectedJobId"
        :key="selectedJobId"
        :job-id="selectedJobId"
        :readonly="chunkReadonly"
        :image-mode="selectedKbIsImageMode"
        @vectorized="onVectorized"
      />
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { docApi } from '../../services/docApi'
import DocList from '../doc/DocList.vue'
import DocSearch from '../doc/DocSearch.vue'
import ChunkEditorPanel from '../doc/ChunkEditorPanel.vue'

const collections = ref([])
const selectedCollection = ref('')
const loading = ref(false)
const docListRef = ref(null)
const chunkDialogVisible = ref(false)
const selectedJobId = ref('')
const chunkReadonly = ref(false)
const selectedKb = computed(() => collections.value.find(col => col.name === selectedCollection.value))
const selectedKbIsImageMode = computed(() =>
  Boolean(selectedKb.value?.image_mode || selectedKb.value?.kb_type === 'multimodal')
)

const openChunkViewer = (jobId, vectorized = false) => {
  if (!jobId) {
    ElMessage.warning('暂无可查看的切片任务')
    return
  }
  selectedJobId.value = jobId
  chunkReadonly.value = Boolean(vectorized)
  chunkDialogVisible.value = true
}

const onVectorized = async () => {
  chunkReadonly.value = true
  await docListRef.value?.load?.()
}

const loadCollections = async () => {
  loading.value = true
  try {
    const res = await docApi.listCollections()
    collections.value = res.data.data?.collections || []
    if (!selectedCollection.value && collections.value.length) {
      selectedCollection.value = collections.value[0].name
    }
  } catch (e) {
    ElMessage.error('加载知识库失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loading.value = false
  }
}

onMounted(loadCollections)
</script>

<style scoped>
.admin-data-view { max-width: 1200px; width: 100%; margin: 0 auto; }
.view-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
:deep(.chunk-view-dialog .el-dialog__body) {
  max-height: 78vh;
  overflow: auto;
}

@media (max-width: 640px) {
  .view-toolbar {
    align-items: stretch;
  }

  .view-toolbar :deep(.el-select),
  .view-toolbar :deep(.el-button) {
    width: 100% !important;
  }
}
</style>
