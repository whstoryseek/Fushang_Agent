<template>
  <el-tabs v-model="activeTab" type="border-card">
    <el-tab-pane label="单文件上传" name="single">
      <el-card shadow="never" style="margin-top: 12px">
        <el-alert
          type="info"
          :closable="false"
          show-icon
          style="margin-bottom: 20px"
          :title="singleUploadTitle"
        />

        <el-form label-width="110px" style="margin-bottom: 16px">
          <el-form-item v-if="!collection" label="目标知识库" required>
            <el-select
              v-model="selectedCollection"
              placeholder="请选择知识库"
              style="width: 320px"
              @focus="loadCollections"
              :loading="collectionsLoading"
              clearable
            >
              <el-option
                v-for="col in collections"
                :key="collectionValue(col)"
                :label="collectionLabel(col)"
                :value="collectionValue(col)"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="切块策略">
            <el-radio-group v-model="config.chunkProfile" size="small">
              <el-radio-button label="smart_mix">智能混合</el-radio-button>
              <el-radio-button label="parent_child">父子块</el-radio-button>
              <el-radio-button label="flat">普通切块</el-radio-button>
            </el-radio-group>
          </el-form-item>

          <el-form-item label="切块预设">
            <el-select
              v-model="config.chunkPreset"
              style="width: 180px"
              @change="applySinglePreset"
            >
              <el-option
                v-for="preset in chunkPresetOptions"
                :key="preset.value"
                :label="preset.label"
                :value="preset.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="切片参数">
            <el-row :gutter="16" style="width: 100%">
              <el-col v-if="config.chunkProfile !== 'flat'" :span="6">
                <el-input-number
                  v-model="config.parentChunkSize"
                  :min="config.childChunkSize"
                  :max="5000"
                  :step="100"
                  style="width: 100%"
                />
                <div class="tip">父块大小</div>
              </el-col>
              <el-col :span="6">
                <el-input-number
                  v-model="config.childChunkSize"
                  :min="100"
                  :max="2048"
                  :step="50"
                  style="width: 100%"
                />
                <div class="tip">子块大小</div>
              </el-col>
              <el-col :span="6">
                <el-input-number
                  v-model="config.chunkOverlap"
                  :min="0"
                  :max="Math.max(config.childChunkSize - 1, 0)"
                  :step="10"
                  style="width: 100%"
                />
                <div class="tip">块重叠</div>
              </el-col>
              <el-col v-if="imageMode" :span="6">
                <el-input-number
                  v-model="config.imageDpi"
                  :min="72"
                  :max="300"
                  :step="50"
                  style="width: 100%"
                />
                <div class="tip">图片 DPI</div>
              </el-col>
            </el-row>
          </el-form-item>
        </el-form>

        <el-upload
          ref="singleUploadRef"
          drag
          :auto-upload="false"
          :limit="1"
          :show-file-list="true"
          :accept="singleAccept"
          :on-change="onSingleFileChange"
        >
          <el-icon style="font-size: 48px"><UploadFilled /></el-icon>
          <div style="margin-top: 8px; font-size: 14px; color: #606266">
            拖拽文件到此处或<em style="color: #409eff; font-style: normal">点击选择</em>
          </div>
          <template #tip>
            <div style="font-size: 12px; color: #909399; margin-top: 4px">
              {{ singleTip }}
            </div>
          </template>
        </el-upload>

        <div style="margin-top: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap">
          <el-switch v-model="syncGraph" active-text="同步知识图谱" inactive-text="仅切片入库" />
          <el-button
            type="primary"
            :loading="singleUploading"
            :disabled="!singleSelectedFile || !resolvedCollection"
            @click="submitSingleUpload"
          >
            开始上传
          </el-button>
        </div>
      </el-card>
    </el-tab-pane>

    <el-tab-pane label="类目批量导入" name="category">
      <el-card shadow="never" style="margin-top: 12px">
        <el-alert
          type="info"
          :closable="false"
          show-icon
          style="margin-bottom: 20px"
          title="选择类目后开始切分，后台会把该类目中的文件按当前知识库配置批量入库。"
        />

        <el-form label-width="120px">
          <el-form-item label="选择类目" required>
            <el-select
              v-model="selectedCategoryId"
              placeholder="请选择类目"
              style="width: 320px"
              @focus="loadCategories"
              :loading="categoriesLoading"
              clearable
            >
              <el-option
                v-for="cat in categories"
                :key="cat.category_id"
                :label="cat.name"
                :value="cat.category_id"
              >
                <span>{{ cat.name }}</span>
                <span style="float: right; color: #909399; font-size: 12px">
                  {{ cat.description || '' }}
                </span>
              </el-option>
            </el-select>
            <el-button size="small" style="margin-left: 8px" @click="$emit('go-categories')">
              管理类目
            </el-button>
          </el-form-item>

          <el-form-item v-if="!collection" label="目标知识库" required>
            <el-select
              v-model="selectedCollection"
              placeholder="请选择知识库"
              style="width: 320px"
              @focus="loadCollections"
              :loading="collectionsLoading"
              clearable
            >
              <el-option
                v-for="col in collections"
                :key="collectionValue(col)"
                :label="collectionLabel(col)"
                :value="collectionValue(col)"
              />
            </el-select>
          </el-form-item>

          <el-divider content-position="left">切片参数</el-divider>

          <el-form-item label="切块策略">
            <el-radio-group v-model="catConfig.chunkProfile" size="small">
              <el-radio-button label="smart_mix">智能混合</el-radio-button>
              <el-radio-button label="parent_child">父子块</el-radio-button>
              <el-radio-button label="flat">普通切块</el-radio-button>
            </el-radio-group>
          </el-form-item>

          <el-form-item label="切块预设">
            <el-select
              v-model="catConfig.chunkPreset"
              style="width: 180px"
              @change="applyCategoryPreset"
            >
              <el-option
                v-for="preset in chunkPresetOptions"
                :key="preset.value"
                :label="preset.label"
                :value="preset.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item v-if="catConfig.chunkProfile !== 'flat'" label="父块大小">
            <el-input-number
              v-model="catConfig.parentChunkSize"
              :min="catConfig.childChunkSize"
              :max="5000"
              :step="100"
              style="width: 180px"
            />
          </el-form-item>

          <el-form-item label="子块大小">
            <el-input-number
              v-model="catConfig.childChunkSize"
              :min="100"
              :max="2048"
              :step="50"
              style="width: 180px"
            />
          </el-form-item>

          <el-form-item label="块重叠">
            <el-input-number
              v-model="catConfig.chunkOverlap"
              :min="0"
              :max="Math.max(catConfig.childChunkSize - 1, 0)"
              :step="10"
              style="width: 180px"
            />
          </el-form-item>

          <el-form-item v-if="imageMode" label="图片 DPI">
            <el-input-number
              v-model="catConfig.imageDpi"
              :min="72"
              :max="300"
              :step="50"
              style="width: 180px"
            />
          </el-form-item>

          <el-form-item label="Excel 每片行数">
            <el-input-number
              v-model="catConfig.excelRowsPerChunk"
              :min="1"
              :max="5000"
              :step="10"
              style="width: 180px"
            />
          </el-form-item>

          <el-form-item label="同步图谱">
            <el-switch v-model="syncGraphCat" />
          </el-form-item>
        </el-form>

        <div style="margin-top: 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap">
          <el-button
            type="primary"
            :disabled="!selectedCategoryId || !resolvedCollection"
            :loading="chunking"
            @click="startChunking"
          >
            开始切分
          </el-button>

          <span v-if="!selectedCategoryId" style="color: #909399; font-size: 13px">
            请先选择类目
          </span>
        </div>

        <el-divider v-if="chunkResult" />
        <el-alert
          v-if="chunkResult"
          :type="chunkResult.errors?.length ? 'warning' : 'success'"
          :closable="false"
          show-icon
          :title="chunkResultTitle"
        />

        <div v-if="chunkResult?.skipped?.length" style="margin-top: 8px">
          <el-alert
            type="warning"
            :closable="false"
            show-icon
            title="以下文件已经存在于知识库中，已自动跳过。"
          />
          <div style="margin-top: 6px; display: flex; flex-direction: column; gap: 4px">
            <el-tag
              v-for="item in chunkResult.skipped"
              :key="item.file_name"
              type="warning"
              size="small"
              style="width: fit-content"
            >
              {{ item.file_name }}
            </el-tag>
          </div>
        </div>
      </el-card>
    </el-tab-pane>

    <el-tab-pane label="Excel 类目上传" name="excel">
      <ExcelCategoryUpload
        :collection="collection"
        @uploaded="$emit('uploaded', $event)"
        @go-categories="$emit('go-categories')"
      />
    </el-tab-pane>
  </el-tabs>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { docApi } from '@/services/docApi'
import ExcelCategoryUpload from './ExcelCategoryUpload.vue'

const emit = defineEmits(['open-jobs', 'uploaded', 'go-categories'])

const props = defineProps({
  collection: { type: String, default: '' },
  imageMode: { type: Boolean, default: false },
})

const collection = computed(() => props.collection)
const imageMode = computed(() => props.imageMode)

const activeTab = ref('single')
const categories = ref([])
const categoriesLoading = ref(false)
const selectedCategoryId = ref('')
const collections = ref([])
const collectionsLoading = ref(false)
const selectedCollection = ref('')
const chunking = ref(false)
const chunkResult = ref(null)
const syncGraph = ref(false)
const syncGraphCat = ref(false)

const CHUNK_PRESETS = {
  precise: { parentChunkSize: 1200, childChunkSize: 350, chunkOverlap: 60 },
  balanced: { parentChunkSize: 1800, childChunkSize: 500, chunkOverlap: 80 },
  broad: { parentChunkSize: 2400, childChunkSize: 700, chunkOverlap: 120 },
}

const chunkPresetOptions = [
  { label: '精细', value: 'precise' },
  { label: '均衡', value: 'balanced' },
  { label: '大上下文', value: 'broad' },
]

const createChunkConfig = (extra = {}) => ({
  chunkProfile: 'smart_mix',
  chunkStrategy: 'parent_child',
  chunkPreset: 'balanced',
  ...CHUNK_PRESETS.balanced,
  ...extra,
})

const config = ref(createChunkConfig({
  imageDpi: 150,
}))

const catConfig = ref(createChunkConfig({
  imageDpi: 150,
  excelRowsPerChunk: 1,
}))

const singleUploadRef = ref(null)
const singleSelectedFile = ref(null)
const singleUploading = ref(false)

const resolvedCollection = computed(() => collection.value || selectedCollection.value)
const singleAccept = computed(() =>
  imageMode.value ? '.pdf,.docx' : '.pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xlsx,.xls'
)
const singleTip = computed(() =>
  imageMode.value
    ? '支持 PDF、DOCX，图片与文本会按统一链路入库，最大 200MB。'
    : '支持 PDF、Word、PPT、TXT、Markdown、Excel，最大 200MB。'
)
const singleUploadTitle = computed(() =>
  imageMode.value
    ? '图文模式单文件上传：PDF 与 DOCX 都会走统一解析链路，保留文本、表格、链接和图片映射。'
    : '标准模式单文件上传：后台自动切分并入库，DOCX 会保留表格和链接文本。'
)
const chunkResultTitle = computed(() => {
  if (!chunkResult.value) {
    return ''
  }
  const skipped = chunkResult.value.skipped?.length || 0
  const errors = chunkResult.value.errors?.length || 0
  return `已提交 ${chunkResult.value.submitted} 个，跳过 ${skipped} 个，失败 ${errors} 个`
})

const collectionLabel = (collection) =>
  collection.display_name || collection.name || collection.collection_name || ''

const collectionValue = (collection) =>
  collection.name || collection.collection_name || ''

const applyPreset = (targetRef) => {
  const preset = CHUNK_PRESETS[targetRef.value.chunkPreset] || CHUNK_PRESETS.balanced
  targetRef.value.parentChunkSize = preset.parentChunkSize
  targetRef.value.childChunkSize = preset.childChunkSize
  targetRef.value.chunkOverlap = preset.chunkOverlap
}

const applySinglePreset = () => applyPreset(config)
const applyCategoryPreset = () => applyPreset(catConfig)

const normalizedChunkPayload = (chunkConfig) => {
  const childSize = Number(chunkConfig.childChunkSize)
  const resolvedStrategy = chunkConfig.chunkProfile === 'smart_mix'
    ? 'parent_child'
    : chunkConfig.chunkProfile
  const parentSize = resolvedStrategy === 'parent_child'
    ? Math.max(Number(chunkConfig.parentChunkSize), childSize)
    : childSize
  const overlap = Math.min(Number(chunkConfig.chunkOverlap), Math.max(childSize - 1, 0))

  return {
    chunk_profile: chunkConfig.chunkProfile,
    chunk_strategy: resolvedStrategy,
    parent_chunk_size: parentSize,
    child_chunk_size: childSize,
    chunk_size: childSize,
    chunk_overlap: overlap,
  }
}

const appendChunkPayload = (formData, chunkConfig) => {
  const payload = normalizedChunkPayload(chunkConfig)
  Object.entries(payload).forEach(([key, value]) => {
    formData.append(key, String(value))
  })
}

const validateFile = (file, allowedExtensions) => {
  if (!file) {
    return false
  }
  const ext = `.${file.name.split('.').pop().toLowerCase()}`
  if (!allowedExtensions.includes(ext)) {
    ElMessage.error(`不支持的文件格式: ${ext}`)
    return false
  }
  if (file.size > 200 * 1024 * 1024) {
    ElMessage.error('文件超过 200MB')
    return false
  }
  return true
}

const onSingleFileChange = (file) => {
  const rawFile = file.raw || file
  const allowed = imageMode.value
    ? ['.pdf', '.docx']
    : ['.pdf', '.doc', '.docx', '.txt', '.md', '.ppt', '.pptx', '.xlsx', '.xls']

  if (!validateFile(rawFile, allowed)) {
    singleSelectedFile.value = null
    singleUploadRef.value?.clearFiles()
    return
  }

  singleSelectedFile.value = rawFile
}

const submitSingleUpload = async () => {
  if (!singleSelectedFile.value) {
    return
  }
  if (!resolvedCollection.value) {
    ElMessage.warning('请先选择目标知识库')
    return
  }

  singleUploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', singleSelectedFile.value)
    formData.append('kb_name', resolvedCollection.value)
    appendChunkPayload(formData, config.value)
    formData.append('image_dpi', String(config.value.imageDpi))
    formData.append('sync_graph', syncGraph.value ? 'true' : 'false')

    const res = await docApi.uploadDocument(formData)
    if (res.data.success) {
      ElMessage.success(res.data.message || '上传成功')
      singleSelectedFile.value = null
      singleUploadRef.value?.clearFiles()
      emit('uploaded', res.data.data)
    }
  } catch (error) {
    ElMessage.error(`上传失败: ${error.response?.data?.detail || error.message}`)
  } finally {
    singleUploading.value = false
  }
}

const loadCategories = async () => {
  if (categoriesLoading.value || categories.value.length > 0) {
    return
  }
  categoriesLoading.value = true
  try {
    const res = await docApi.listCategories()
    categories.value = res.data.data.categories || []
  } catch (error) {
    console.error(error)
  } finally {
    categoriesLoading.value = false
  }
}

const loadCollections = async () => {
  if (collectionsLoading.value || collections.value.length > 0) {
    return
  }
  collectionsLoading.value = true
  try {
    const res = await docApi.listCollections()
    collections.value = res.data.data?.collections || []
  } catch (error) {
    console.error(error)
  } finally {
    collectionsLoading.value = false
  }
}

const startChunking = async () => {
  if (!selectedCategoryId.value) {
    return
  }
  if (!resolvedCollection.value) {
    ElMessage.warning('请先选择目标知识库')
    return
  }

  chunking.value = true
  chunkResult.value = null
  try {
    const res = await docApi.startChunking(selectedCategoryId.value, {
      kb_name: resolvedCollection.value,
      ...normalizedChunkPayload(catConfig.value),
      image_dpi: catConfig.value.imageDpi,
      sync_graph: syncGraphCat.value,
      excel_rows_per_chunk: catConfig.value.excelRowsPerChunk,
    })
    chunkResult.value = res.data.data

    const { submitted, errors } = res.data.data
    if (submitted === 0) {
      ElMessage.info(res.data.message)
    } else {
      ElMessage.success(`已提交 ${submitted} 个文件切分任务`)
      emit('uploaded', res.data.data)
    }
    if (errors?.length) {
      ElMessage.warning(`${errors.length} 个文件提交失败`)
    }
  } catch (error) {
    ElMessage.error(`切分失败: ${error.response?.data?.detail || error.message}`)
  } finally {
    chunking.value = false
  }
}
</script>

<style scoped>
.tip {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
}

:deep(.el-upload-dragger) {
  padding: 30px;
}
</style>
