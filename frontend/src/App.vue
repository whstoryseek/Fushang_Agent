<template>
  <div id="app">
    <!-- Aurora background blobs -->
    <div class="aurora-bg" aria-hidden="true">
      <div class="blob blob-1" />
      <div class="blob blob-2" />
      <div class="blob blob-3" />
      <div class="noise" />
    </div>

    <div class="app-layout">
      <!-- Icon sidebar -->
      <aside class="sidebar">
        <div class="sidebar-logo">
          <div class="logo-mark" style="font-size:12px; font-weight:700;">
            天合
          </div>
        </div>

        <nav class="nav-list">
          <div class="nav-section-title">问答</div>
          <el-tooltip v-for="item in qaNavItems" :key="item.key"
            :content="item.label" placement="right" effect="dark">
            <button
              class="nav-item"
              :class="{ active: activeMenu === item.key }"
              @click="handleMenuSelect(item.key)"
            >
              <el-icon><component :is="item.icon" /></el-icon>
              <span v-if="activeMenu === item.key" class="nav-active-dot" />
            </button>
          </el-tooltip>

          <div v-if="isAdmin" class="nav-section-title" style="margin-top:12px;">管理</div>
          <el-tooltip v-for="item in visibleAdminNavItems" :key="item.key"
            :content="item.label" placement="right" effect="dark">
            <button
              class="nav-item"
              :class="{ active: activeMenu === item.key || (item.match && activeMenu.startsWith(item.match)) }"
              @click="handleMenuSelect(item.key)"
            >
              <el-icon><component :is="item.icon" /></el-icon>
              <span v-if="activeMenu === item.key || (item.match && activeMenu.startsWith(item.match))" class="nav-active-dot" />
            </button>
          </el-tooltip>
        </nav>
      </aside>

      <!-- Main -->
      <div class="main-wrap">
        <!-- Topbar -->
        <header class="topbar">
          <div class="topbar-left">
            <span class="page-title">{{ pageTitle }}</span>
            <span v-if="pageSubtitle" class="page-subtitle">{{ pageSubtitle }}</span>
          </div>
          <div class="topbar-right">
            <div class="model-select-wrap">
              <el-icon class="model-icon"><cpu /></el-icon>
              <el-select v-model="selectedModel" size="small" style="width:160px" placeholder="模型">
                <el-option v-for="m in availableModels" :key="m.name" :label="m.name" :value="m.name" />
              </el-select>
            </div>
            <el-button v-if="!isAdmin" size="small" plain @click="loginDialogVisible = true">
              管理员登录
            </el-button>
            <div v-else class="admin-session">
              <el-tag size="small" type="success">{{ currentAdmin?.username || 'admin' }}</el-tag>
              <el-button size="small" plain @click="logoutAdmin">退出</el-button>
            </div>
            <div class="status-pill" :class="apiStatus ? 'online' : 'offline'">
              <span class="pulse-dot" />
              {{ apiStatus ? 'Connected' : 'Offline' }}
            </div>
          </div>
        </header>

        <!-- Content -->
        <main class="content">
          <div v-show="activeMenu === 'chat'"><SimpleChat :model="selectedModel" :is-admin="isAdmin" /></div>
          <div v-show="activeMenu === 'user-history'"><UserHistory /></div>
          <template v-if="isAdmin">
            <div v-show="activeMenu === 'admin-data-import'"><AdminDataImport :collection="selectedCollection" /></div>
            <div v-show="activeMenu === 'admin-data-view'"><AdminDataView /></div>
            <div v-show="activeMenu === 'admin-unanswered'"><AdminUnanswered /></div>
            <div v-show="activeMenu.startsWith('admin')"><AdminPanel :active-tab="adminTab" /></div>
          </template>
        </main>
      </div>
    </div>

    <el-dialog v-model="loginDialogVisible" title="管理员登录" width="360px" destroy-on-close>
      <el-form :model="loginForm" label-width="70px" @keydown.enter.prevent="loginAdmin">
        <el-form-item label="账号">
          <el-input v-model="loginForm.username" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="loginForm.password" type="password" autocomplete="current-password" show-password />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="loginDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="loginLoading" @click="loginAdmin">登录</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { AUTH_EXPIRED_EVENT, authService, getStoredAdmin } from './services/auth'
import { apiService } from './services/api'
import SimpleChat from './components/SimpleChat.vue'
import AdminPanel from './components/AdminPanel.vue'
import UserHistory from './components/UserHistory.vue'
import AdminDataImport from './components/admin/AdminDataImport.vue'
import AdminDataView from './components/admin/AdminDataView.vue'
import AdminUnanswered from './components/admin/AdminUnanswered.vue'

const activeMenu = ref('chat')
const selectedModel = ref('qwen-turbo')
const selectedCollection = ref('')
const availableModels = ref([])
const apiStatus = ref(false)
const currentAdmin = ref(getStoredAdmin())
const isAdmin = computed(() => currentAdmin.value?.role === 'admin')
const loginDialogVisible = ref(false)
const loginLoading = ref(false)
const loginForm = ref({ username: 'admin', password: '' })

const qaNavItems = [
  { key: 'chat', label: '智能问答', icon: 'ChatDotRound' },
  { key: 'user-history', label: '对话历史', icon: 'Clock' },
]

const adminNavItems = [
  { key: 'admin-data-import', label: '数据导入', icon: 'UploadFilled' },
  { key: 'admin-data-view', label: '数据查看', icon: 'View' },
  { key: 'admin-unanswered', label: '未回答问题', icon: 'WarningFilled' },
  { key: 'admin-collections', label: '系统设置', icon: 'Setting', match: 'admin' },
]
const visibleAdminNavItems = computed(() => isAdmin.value ? adminNavItems : [])

const adminTabMap = { 'admin-collections': 'collections', 'admin-create': 'create', 'admin-config': 'config' }
const adminTab = computed(() => adminTabMap[activeMenu.value] || 'collections')

const pageMeta = {
  chat:               { title: '智能问答', sub: '天合人康扶商问答系统' },
  'user-history':     { title: '对话历史', sub: '查看您的问答记录' },
  'admin-data-import':{ title: '数据导入', sub: '导入文档到知识库' },
  'admin-data-view':  { title: '数据查看', sub: '查看知识库数据' },
  'admin-unanswered': { title: '未回答问题', sub: 'AI 未能回答的问题监控' },
  'admin-collections':{ title: '系统设置', sub: '知识库配置与管理' },
  'admin-create':     { title: '创建知识库', sub: '新建向量集合' },
  'admin-config':     { title: '配置信息', sub: '系统参数' },
}
const pageTitle    = computed(() => pageMeta[activeMenu.value]?.title || '')
const pageSubtitle = computed(() => pageMeta[activeMenu.value]?.sub || '')

const handleMenuSelect = (key) => {
  if (key.startsWith('admin') && !isAdmin.value) {
    loginDialogVisible.value = true
    return
  }
  activeMenu.value = key
}
const handleResumeSession = () => { activeMenu.value = 'chat' }

const loginAdmin = async () => {
  if (!loginForm.value.username || !loginForm.value.password) {
    ElMessage.warning('请输入管理员账号和密码')
    return
  }
  loginLoading.value = true
  try {
    currentAdmin.value = await authService.login(loginForm.value.username, loginForm.value.password)
    loginDialogVisible.value = false
    loginForm.value.password = ''
    ElMessage.success('管理员已登录')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '登录失败')
  } finally {
    loginLoading.value = false
  }
}

const logoutAdmin = async () => {
  await authService.logout()
  currentAdmin.value = null
  if (activeMenu.value.startsWith('admin')) activeMenu.value = 'chat'
  ElMessage.success('已退出管理员登录')
}

const handleAuthExpired = () => {
  currentAdmin.value = null
  if (activeMenu.value.startsWith('admin')) activeMenu.value = 'chat'
  ElMessage.warning('管理员登录已失效，请重新登录')
}

onMounted(async () => {
  window.addEventListener('knowledge-session:resume', handleResumeSession)
  window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired)
  try {
    const user = await authService.me()
    if (user) currentAdmin.value = user
  } catch {}
  try {
    const res = await apiService.getModels()
    availableModels.value = res.models
    selectedModel.value = res.default_model
    apiStatus.value = true
  } catch {
    availableModels.value = [
      { name: 'doubao-seed-2-0-pro-260215' }
    ]
  }
})
onUnmounted(() => {
  window.removeEventListener('knowledge-session:resume', handleResumeSession)
  window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired)
})
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; }
</style>

<style scoped>
/* ── Aurora background ── */
.aurora-bg {
  position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
}
.blob {
  position: absolute; border-radius: 50%;
  filter: blur(90px); opacity: 0.28;
  animation: drift 20s ease-in-out infinite alternate;
}
.blob-1 {
  width: 650px; height: 650px; top: -220px; left: -120px;
  background: radial-gradient(circle, #4f8ef7 0%, #7c3aed 50%, transparent 100%);
  animation-duration: 22s;
}
.blob-2 {
  width: 550px; height: 550px; bottom: -180px; right: -120px;
  background: radial-gradient(circle, #06b6d4 0%, #34d399 50%, transparent 100%);
  animation-duration: 18s; animation-delay: -8s;
}
.blob-3 {
  width: 480px; height: 480px; top: 35%; left: 45%;
  background: radial-gradient(circle, #a78bfa 0%, #f472b6 50%, transparent 100%);
  animation-duration: 25s; animation-delay: -14s; opacity: 0.22;
}
.noise {
  position: absolute; inset: 0; opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  background-size: 200px 200px;
}
@keyframes drift {
  0%   { transform: translate(0, 0) scale(1); }
  33%  { transform: translate(40px, -30px) scale(1.05); }
  66%  { transform: translate(-20px, 40px) scale(0.97); }
  100% { transform: translate(30px, 20px) scale(1.03); }
}

/* ── Layout ── */
#app { height: 100vh; height: 100dvh; overflow: hidden; }
.app-layout {
  position: relative; z-index: 1;
  display: flex; height: 100vh; height: 100dvh; overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 60px; flex-shrink: 0;
  display: flex; flex-direction: column; align-items: center;
  background: rgba(13,17,23,0.75);
  backdrop-filter: blur(28px);
  border-right: 1px solid rgba(255,255,255,0.08);
  padding: 0;
}
.sidebar-logo {
  height: 60px; display: flex; align-items: center; justify-content: center;
  width: 100%; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.logo-mark {
  width: 34px; height: 34px; border-radius: 10px;
  background: linear-gradient(135deg, #3b6fd4, #5b4fcf);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: #fff;
  box-shadow: 0 4px 14px rgba(59,111,212,0.5);
}
.nav-list {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; padding: 8px 0; gap: 2px; width: 100%;
  overflow-y: auto;
}
.nav-section-title {
  font-size: 10px; color: rgba(255,255,255,0.18);
  text-transform: uppercase; letter-spacing: 0.8px;
  padding: 8px 0 4px; width: 100%; text-align: center;
}
.nav-item {
  position: relative;
  width: 40px; height: 40px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: rgba(255,255,255,0.3); font-size: 17px;
  background: transparent; border: none; outline: none;
  transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
}
.nav-item:hover { background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.7); }
.nav-item.active {
  background: rgba(79,142,247,0.15);
  color: #7eb3ff;
  box-shadow: 0 0 0 1px rgba(79,142,247,0.3);
}
.nav-active-dot {
  position: absolute; right: -1px; top: 50%; transform: translateY(-50%);
  width: 3px; height: 16px; border-radius: 99px;
  background: linear-gradient(180deg, #7eb3ff, #a78bfa);
  box-shadow: 0 0 8px rgba(79,142,247,0.8);
}
.sidebar-bottom {
  padding: 12px 0; border-top: 1px solid rgba(255,255,255,0.05);
  width: 100%; display: flex; justify-content: center;
}

/* ── Main wrap ── */
.main-wrap {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
}

/* ── Topbar ── */
.topbar {
  height: 60px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 28px;
  background: rgba(13,17,23,0.65);
  backdrop-filter: blur(28px);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.topbar-left { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
.page-title { font-size: 15px; font-weight: 700; color: #f0f4ff; letter-spacing: -0.2px; }
.page-subtitle { font-size: 12px; color: rgba(255,255,255,0.25); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.topbar-right { display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
.model-select-wrap { display: flex; align-items: center; gap: 6px; min-width: 0; }
.model-select-wrap :deep(.el-select) { max-width: 100%; }
.model-icon { color: rgba(255,255,255,0.3); font-size: 14px; }
.admin-session { display: flex; align-items: center; gap: 8px; }

.status-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 99px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.3px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.4);
  max-width: 100%;
}
.status-pill.online { color: #2dd4a0; border-color: rgba(45,212,160,0.25); background: rgba(45,212,160,0.06); }
.pulse-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: rgba(255,255,255,0.3);
}
.status-pill.online .pulse-dot {
  background: #2dd4a0;
  box-shadow: 0 0 0 0 rgba(45,212,160,0.4);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(45,212,160,0.4); }
  70%  { box-shadow: 0 0 0 6px rgba(45,212,160,0); }
  100% { box-shadow: 0 0 0 0 rgba(45,212,160,0); }
}

/* ── Content ── */
.content {
  flex: 1; overflow-y: auto; padding: 28px 32px;
}
.content > div { min-width: 0; }
.content::-webkit-scrollbar { width: 4px; }
.content::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 99px; }

@media (max-width: 900px) {
  .topbar {
    height: auto;
    min-height: 60px;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 16px;
  }

  .topbar-left {
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    flex: 1 1 220px;
  }

  .topbar-right {
    flex: 1 1 320px;
    flex-wrap: wrap;
  }
}

@media (max-width: 768px) {
  .blob {
    filter: blur(70px);
    opacity: 0.2;
  }

  .blob-1 { width: 360px; height: 360px; top: -120px; left: -120px; }
  .blob-2 { width: 340px; height: 340px; bottom: 20px; right: -160px; }
  .blob-3 { width: 280px; height: 280px; top: 35%; left: 35%; }

  .app-layout {
    flex-direction: column;
  }

  .main-wrap {
    order: 1;
    height: calc(100dvh - 64px);
    min-height: 0;
  }

  .sidebar {
    order: 2;
    width: 100%;
    height: 64px;
    flex-direction: row;
    justify-content: center;
    border-right: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding: 0 max(10px, env(safe-area-inset-right)) env(safe-area-inset-bottom) max(10px, env(safe-area-inset-left));
  }

  .sidebar-logo,
  .nav-section-title,
  .sidebar-bottom {
    display: none;
  }

  .nav-list {
    flex: 0 1 auto;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 0;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .nav-item {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    flex: 0 0 auto;
  }

  .nav-active-dot {
    right: 50%;
    top: auto;
    bottom: -2px;
    width: 18px;
    height: 3px;
    transform: translateX(50%);
    background: linear-gradient(90deg, #7eb3ff, #a78bfa);
  }

  .topbar {
    padding: 10px 12px;
  }

  .topbar-right {
    justify-content: flex-start;
    gap: 8px;
    width: 100%;
  }

  .model-select-wrap {
    flex: 1 1 180px;
  }

  .model-select-wrap :deep(.el-select) {
    width: 100% !important;
  }

  .admin-session {
    flex-wrap: wrap;
  }

  .status-pill {
    margin-left: auto;
  }

  .content {
    display: flex;
    flex-direction: column;
    padding: 14px 12px;
    min-height: 0;
    overflow-x: hidden;
  }

  .content > div {
    flex: 1;
    width: 100%;
    min-height: 0;
    min-width: 0;
  }
}

@media (max-width: 480px) {
  .topbar-left {
    flex-basis: 100%;
  }

  .page-subtitle {
    max-width: 100%;
  }

  .topbar-right {
    flex-basis: 100%;
    width: 100%;
  }

  .model-select-wrap {
    flex-basis: 100%;
  }

  .topbar-right :deep(.el-button) {
    flex: 1 1 auto;
  }

  .status-pill {
    margin-left: 0;
    flex: 1 1 100%;
    justify-content: center;
  }
}
</style>
