<template>
  <div class="admin-user-management">
    <section class="hero-card">
      <div>
        <p class="eyebrow">主管理员操作</p>
        <h2>分发子管理员账号</h2>
        <p class="hero-copy">
          新建子管理员后，系统默认密码为 <strong>88888888</strong>。建议分发后尽快通知对方修改密码。
        </p>
      </div>
      <div class="hero-badge">仅主管理员可见</div>
    </section>

    <section class="panel-card">
      <div class="panel-head">
        <div>
          <h3>创建账号</h3>
          <p>输入子管理员账号名并提交。</p>
        </div>
      </div>

      <el-form class="create-form" @submit.prevent="createSubAdmin">
        <el-form-item label="账号名">
          <el-input
            v-model="form.username"
            placeholder="例如：ops01"
            maxlength="50"
            clearable
            @keydown.enter.prevent="createSubAdmin"
          />
        </el-form-item>
        <div class="form-actions">
          <el-button type="primary" :loading="creating" @click="createSubAdmin">创建子管理员</el-button>
          <span class="password-tip">默认密码：88888888</span>
        </div>
      </el-form>
    </section>

    <section class="panel-card">
      <div class="panel-head">
        <div>
          <h3>已有管理员</h3>
          <p>查看当前主管理员和子管理员账号。</p>
        </div>
        <el-button plain :loading="loading" @click="loadUsers">刷新</el-button>
      </div>

      <el-table v-loading="loading" :data="users" stripe class="users-table">
        <el-table-column prop="username" label="账号" min-width="180" />
        <el-table-column label="类型" width="140">
          <template #default="{ row }">
            <el-tag :type="row.role === 'super_admin' ? 'danger' : 'success'">
              {{ row.role === 'super_admin' ? '主管理员' : '子管理员' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'info'" effect="plain">
              {{ row.is_active ? '启用' : '停用' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </section>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { apiService } from '../../services/api'

const loading = ref(false)
const creating = ref(false)
const users = ref([])
const form = ref({ username: '' })

const loadUsers = async () => {
  loading.value = true
  try {
    const response = await apiService.listAdminUsers()
    users.value = response.data?.users || []
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '加载管理员列表失败')
  } finally {
    loading.value = false
  }
}

const createSubAdmin = async () => {
  const username = form.value.username.trim()
  if (!username) {
    ElMessage.warning('请输入子管理员账号')
    return
  }

  creating.value = true
  try {
    const response = await apiService.createSubAdmin(username)
    const defaultPassword = response.data?.default_password || '88888888'
    ElMessage.success(`创建成功，默认密码：${defaultPassword}`)
    form.value.username = ''
    await loadUsers()
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '创建子管理员失败')
  } finally {
    creating.value = false
  }
}

onMounted(loadUsers)
</script>

<style scoped>
.admin-user-management {
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  gap: 18px;
}

.hero-card,
.panel-card {
  border: 1px solid rgba(191, 208, 224, 0.65);
  background:
    linear-gradient(160deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.82)),
    rgba(255, 255, 255, 0.88);
  border-radius: 22px;
  backdrop-filter: blur(18px);
  box-shadow: 0 18px 50px rgba(64, 85, 106, 0.08);
}

.hero-card {
  padding: 24px 26px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.eyebrow {
  margin: 0 0 10px;
  color: #4f8ef7;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-card h2,
.panel-head h3 {
  margin: 0;
  color: #4d6174;
}

.hero-copy,
.panel-head p,
.password-tip {
  color: #7d8d9c;
}

.hero-copy {
  margin: 10px 0 0;
  line-height: 1.7;
}

.hero-badge {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(79, 142, 247, 0.1);
  border: 1px solid rgba(79, 142, 247, 0.18);
  color: #4f8ef7;
  font-size: 12px;
  white-space: nowrap;
}

.panel-card {
  padding: 22px 24px;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  margin-bottom: 18px;
}

.panel-head p {
  margin: 8px 0 0;
}

.create-form :deep(.el-form-item__label) {
  color: #5f7285;
}

.form-actions {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.users-table {
  width: 100%;
}

.users-table :deep(.el-table),
.users-table :deep(.el-table__inner-wrapper),
.users-table :deep(.el-table tr),
.users-table :deep(.el-table th.el-table__cell),
.users-table :deep(.el-table td.el-table__cell) {
  background: transparent;
}

.users-table :deep(.el-table th.el-table__cell) {
  color: #7d8d9c;
}

.users-table :deep(.el-table td.el-table__cell) {
  color: #4d6174;
}

@media (max-width: 768px) {
  .hero-card,
  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .panel-card,
  .hero-card {
    padding: 18px;
  }
}
</style>
