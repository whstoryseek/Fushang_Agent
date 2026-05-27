# 前端模板拆分迁移设计

## 目标

在不改动后端 API 形态、后端功能逻辑和后端业务代码的前提下：

- 用新的模板风格重做当前用户问答前端
- 单独建设一个管理员前端
- 两个新前端都接入现有后端接口
- 迁移期间保留现有 `frontend/` 作为回退方案

本次任务的本质是“前端替换与拆分”，不是“系统后端重构”。

已确认的硬约束如下：

- 后端接口地址、请求方法、请求体字段、响应体字段都不能改
- 后端 Python 代码和功能逻辑都不能改
- 用户端现有能力要全部保留
- 管理端现有能力要全部保留
- 两个新前端都放在当前仓库内
- 旧的混合前端暂时保留，直到两个新前端验收通过

## 现状背景

当前仓库里的前端是一个用户端和管理端混合在一起的 Vue 应用，位于 `frontend/`：

- `frontend/src/App.vue`
  - 当前总入口，同时挂载问答页、历史页和管理员各模块
- `frontend/src/components/SimpleChat.vue`
  - 当前用户端知识库问答主界面
  - 包含会话侧边栏、流式回答、图片提问、来源展示等能力
- `frontend/src/components/UserHistory.vue`
  - 当前用户端历史记录页面
- `frontend/src/components/AdminPanel.vue`
  - 当前管理端知识库列表、创建、配置等能力
- `frontend/src/components/admin/AdminDataImport.vue`
  - 当前管理端数据导入能力
- `frontend/src/components/admin/AdminDataView.vue`
  - 当前管理端数据查看能力
- `frontend/src/components/admin/AdminServiceTickets.vue`
  - 当前管理端服务记录/工单能力
- `frontend/src/services/api.js`
  - 当前 `/api/v1/*` 接口调用封装，覆盖用户问答与部分管理端接口
- `frontend/src/services/auth.js`
  - 当前管理员登录、登出、`me`、token 持久化以及游客请求头逻辑
- `frontend/src/utils/entryIdentity.mjs`
  - 当前门店/入口身份透传逻辑

你提供的前端模板位于：

`C:/Users/story/Desktop/扶商智能体知识库/前端重构/projects`

这是一个偏用户端的 UI 模板工程，技术栈是：

- Vue 3
- TypeScript
- Vite
- Tailwind CSS v4

模板当前主要包含这些 UI 骨架组件：

- `src/App.vue`
- `src/components/AppHeader.vue`
- `src/components/AppSidebar.vue`
- `src/components/ChatArea.vue`
- `src/components/WelcomeScreen.vue`
- `src/components/InputArea.vue`

模板已经明确了目标视觉语言：

- 留白充足
- 浅色背景
- 靛蓝色主色
- 柔和圆角
- 轻量交互反馈
- 不走传统厚重后台风格

管理员前端也要沿用这套气质，而不是做成传统深色、厚边框、高密度的管理台。

## 推荐方案

在当前仓库中新增两个独立前端项目：

- `frontend-user/`
  - 面向最终用户的问答前端
- `frontend-admin/`
  - 面向管理员的后台前端

两个新前端都继续直接调用现有后端的 `/api/v1/*` 接口。

这次迁移遵循“稳定优先、低风险优先”的原则：

- 本阶段不强行抽共享 npm 包
- 两个新前端各自保留自己的 service 层
- 可以从旧前端复制稳定逻辑并按需整理
- 先保证功能对齐和接口对齐，再考虑后续代码抽象

这样做的原因是：

- 你明确要求不能动后端
- 当前旧前端已经将用户端和管理端混在一起
- 当前工作区还存在未提交变更
- 先独立落地两个新前端，比顺手做大型工程重构更稳

## 目标架构

### 应用拆分

`frontend-user/` 负责：

- 知识库问答
- 流式回答展示
- 会话列表与会话恢复
- 图片提问
- 用户历史记录
- 门店/入口身份透传

`frontend-admin/` 负责：

- 管理员登录与登录态保持
- 知识库列表与管理
- 知识库创建与检索配置
- 数据导入
- 数据查看
- 服务记录/工单处理
- 当前管理端已经暴露的文档、chunk、类目、图谱等工具能力

旧的 `frontend/` 在迁移期间继续保留，承担三种作用：

- 作为接口契约对照物
- 作为行为对照物
- 作为回退方案

### 用户前端结构

`frontend-user/` 以模板工程为视觉基础，在此之上接入当前真实业务状态。建议结构如下：

```text
frontend-user/
  src/
    App.vue
    main.ts
    style.css
    components/
      AppHeader.vue
      AppSidebar.vue
      ChatArea.vue
      WelcomeScreen.vue
      InputArea.vue
      SessionList.vue
      MessageBubble.vue
      SourcePanel.vue
      QueryImageTray.vue
    views/
      ChatView.vue
      HistoryView.vue
    services/
      api.ts
      auth.ts
    utils/
      entryIdentity.ts
      conversationParams.ts
```

模板组件的职责演进如下：

- `AppSidebar.vue`
  - 从模板中的假历史记录，演进为真实会话列表和搜索入口
- `ChatArea.vue`
  - 从模板中的假消息流，演进为真实对话渲染、流式增量、来源信息、置信信息和图片展示
- `InputArea.vue`
  - 从模板中的演示输入框，演进为真实文本发送、图片上传、回车发送、禁用态与错误态处理
- `WelcomeScreen.vue`
  - 保留为空状态欢迎页和快捷提问入口

用户前端中不再包含管理员登录和管理员导航。

### 管理员前端结构

`frontend-admin/` 保持同一套视觉气质，但页面结构要针对后台操作进行组织。建议结构如下：

```text
frontend-admin/
  src/
    App.vue
    main.ts
    style.css
    components/
      AdminShell.vue
      AdminHeader.vue
      AdminSidebar.vue
      AdminLoginDialog.vue
      AdminPageCard.vue
      AdminSectionHeader.vue
    views/
      CollectionsView.vue
      CreateCollectionView.vue
      ConfigView.vue
      DataImportView.vue
      DataViewView.vue
      ServiceTicketsView.vue
    components/
      doc/
      admin/
    services/
      api.ts
      auth.ts
    utils/
      adminNavigation.ts
      adminConfig.ts
```

管理员前端要保留现有全部后台能力，但展示方式改为独立后台壳：

- 独立管理员登录流程
- 独立管理员侧边栏
- 顶部状态栏与管理员身份展示
- 页面级卡片、表格、抽屉、表单统一采用模板风格表达

管理员前端的视觉优先级必须服从模板气质：

- 浅色背景
- 轻边框
- 柔和圆角
- 节制的状态色
- 避免厚重、强对比、传统企业后台风

## 功能映射

### 用户端功能映射

`frontend-user/` 需要完整承接以下旧能力：

- `frontend/src/components/SimpleChat.vue`
  - 知识库问答主流程
  - 知识库选择逻辑
  - `/api/v1/knowledge/stream` 的 SSE 消费逻辑
  - 来源信息展开
  - 图片提问上传与预览
  - 会话创建、切换、删除、恢复
- `frontend/src/components/UserHistory.vue`
  - 用户历史记录页
  - 从历史记录恢复会话
- `frontend/src/services/api.js`
  - `knowledgeQuery`
  - `knowledgeQueryStream`
  - `listCollections`
  - 如果用户端依赖图片解析辅助接口，也要原样承接
- `frontend/src/utils/entryIdentity.mjs`
  - 游客/门店入口身份识别
  - 请求头透传逻辑

当前混在用户壳里的管理员功能，都要从用户前端移除。

### 管理端功能映射

`frontend-admin/` 需要完整承接以下旧能力：

- `frontend/src/services/auth.js`
  - 登录
  - 登出
  - `me`
  - token 持久化
  - 登录过期处理
- `frontend/src/components/AdminPanel.vue`
  - 知识库列表
  - 知识库创建
  - 检索配置
  - 配置信息展示
- `frontend/src/components/admin/AdminDataImport.vue`
  - 文档上传
  - Excel 类目上传
- `frontend/src/components/admin/AdminDataView.vue`
  - 按知识库查看数据
  - 关联的 chunk/图谱/文档视图
- `frontend/src/components/admin/AdminServiceTickets.vue`
  - 服务记录/工单列表
  - 筛选、统计、详情、编辑、chunk 修补、重新向量化
- `frontend/src/components/doc/*`
  - 当前后台依赖的全部文档管理子流程

本次迁移不允许删减任何管理员能力。

## 数据流与接口接入原则

两个新前端都必须严格保持现有后端契约不变。

### 通用规则

- `/api/v1/*` 基础路径不变
- 请求方法不变
- 请求体字段名不变
- 响应消费方式不随意改语义
- 鉴权头逻辑不变
- 游客/门店入口头部透传逻辑不变
- SSE 事件消费协议不变
- 现有可选字段容错方式不变

### 用户端请求流

用户提问时：

- 前端支持纯文本提问
- 前端支持文本加图片提问
- 发送到 `/api/v1/knowledge` 或 `/api/v1/knowledge/stream` 的 payload 结构必须与旧前端一致
- SSE 解析继续保留 `meta`、`delta`、`done`、`error` 的处理方式
- UI 只重做展示，不改变服务端流式行为

用户历史与会话相关流程：

- 会话 ID、恢复逻辑和相关前端状态机制要与旧前端行为保持一致
- 如果旧前端对某些本地状态有特殊约定，应优先复用相同行为

### 管理端请求流

管理员登录相关：

- 登录继续调用 `/api/v1/auth/login`
- 当前用户继续调用 `/api/v1/auth/me`
- 登出继续调用 `/api/v1/auth/logout`

管理员业务相关：

- 知识库集合继续使用 `/api/v1/admin/collections`
- 服务记录/工单继续使用现有 `/api/v1/admin/service-tickets*`
- 文档、chunk、导入、删除、更新等继续走现有接口

管理端可以重组页面，但不能擅自重新定义后端返回值的业务含义。

## 样式与交互规则

模板工程定义了两个新前端共同遵循的视觉基线。

必须保持的样式原则：

- 视觉层级主要依赖留白、背景层次和字重，而不是粗边框
- 主操作色保持靛蓝系
- 背景保持明亮、干净、通透
- 文字风格简洁克制
- 动效和反馈轻量，不炫技

管理端的落地解释规则：

- 表格放在柔和卡片容器中
- 行 hover、选中态要轻
- 表单按逻辑分组，拉开呼吸感
- 抽屉、详情面板、弹窗要延续同一产品家族气质
- 成功、警告、错误等状态色只做清晰表达，不做强刺激

## 错误处理

本次迁移不应掩盖后端错误，但可以把错误展示做得更清楚。

用户端需要保留：

- 流式请求失败处理
- 超时处理
- 空知识库或不可用知识库提示
- 当前前端已存在的图片重置或前端校验行为

管理端需要保留：

- 登录过期处理
- 页面加载失败提示
- 知识库、工单、文档相关变更失败提示
- 当前旧前端已支持的刷新或重试能力

任何增强都只能停留在前端展示层，不能偷偷引入新的后端前提。

## 验证策略

迁移完成后要从三个层面验证。

### 契约对齐验证

需要验证：

- 旧用户端每个关键请求，在 `frontend-user/` 中都有对应实现
- 旧管理端每个关键请求，在 `frontend-admin/` 中都有对应实现
- 鉴权头、游客头、门店入口头没有丢失
- SSE 解析行为和旧前端功能等价

### 功能对齐验证

用户端验证项：

- 知识库提问
- 流式回答展示
- 会话切换
- 会话删除
- 历史页跳转与恢复
- 图片提问
- 门店/入口身份透传

管理端验证项：

- 登录、刷新后保持登录、登出
- 知识库列表与创建
- 检索配置/配置查看
- 数据导入
- 数据查看
- 服务记录/工单筛选、详情、修改、重新向量化
- 文档与 chunk 子流程

### 构建验证

两个新前端都必须满足：

- 可以独立安装依赖
- 可以在本地开发模式下接现有后端运行
- 可以独立完成生产构建

旧的 `frontend/` 在新前端验收前继续保留，作为对照和回退方案。

## 范围边界

本次范围内：

- 新建 `frontend-user/`
- 新建 `frontend-admin/`
- 两个新前端接入当前后端
- 迁移用户端全部现有能力
- 迁移管理端全部现有能力
- 将模板视觉语言扩展到用户端与管理端

本次明确不在范围内：

- 修改后端 Python 代码
- 修改后端接口形态
- 修改后端业务逻辑
- 为了代码“好看”而额外引入共享前端包
- 第一轮迁移时删除旧 `frontend/`
- 借这次任务顺手改产品需求

## 成功标准

满足以下条件即视为成功：

- 仓库中存在两个可独立运行的前端：`frontend-user/` 和 `frontend-admin/`
- `frontend-user/` 在视觉上贴近提供的模板，同时完整保留当前用户端能力
- `frontend-admin/` 在视觉上延续同一模板语言，同时完整保留当前管理端能力
- 两个新前端都直接复用现有后端接口，不需要后端改代码
- 旧 `frontend/` 在新前端确认通过前仍可作为回退方案存在
