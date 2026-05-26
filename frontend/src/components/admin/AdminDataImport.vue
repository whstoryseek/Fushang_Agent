<template>
  <div class="admin-data-import">
    <el-tabs type="border-card">
      <el-tab-pane label="文档导入" name="doc">
        <DocUpload :collection="collection" />
      </el-tab-pane>
      <el-tab-pane label="Excel 导入" name="excel">
        <ExcelCategoryUpload :collection="collection" />
      </el-tab-pane>
      <el-tab-pane label="类目管理" name="category">
        <CategoryManager />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import DocUpload from '../doc/DocUpload.vue'
import ExcelCategoryUpload from '../doc/ExcelCategoryUpload.vue'
import CategoryManager from '../doc/CategoryManager.vue'

defineProps({
  collection: { type: String, default: '' },
})
</script>

<style scoped>
.admin-data-import { max-width: 1200px; width: 100%; margin: 0 auto; }

@media (max-width: 768px) {
  .admin-data-import :deep(.el-tabs__nav-scroll) {
    overflow-x: auto;
  }

  .admin-data-import :deep(.el-tabs__item) {
    padding: 0 14px;
  }
}
</style>
