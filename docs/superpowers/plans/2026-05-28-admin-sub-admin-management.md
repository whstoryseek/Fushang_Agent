# Admin Sub-Admin Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-level admin model where the seeded main admin becomes `super_admin`, sub-admins can log in and use the existing admin area, and only the main admin can create sub-admin accounts with default password `88888888`.

**Architecture:** Keep the existing `auth_user` table and JWT flow, but expand `role` to `super_admin` and `sub_admin`. Reuse the current `/api/v1/admin/*` namespace, adding a focused user-management router protected by a new `require_super_admin` dependency. In the frontend, keep the existing admin shell and add a minimal "管理员账号" page visible only to `super_admin`.

**Tech Stack:** FastAPI, repository-based PostgreSQL access, unittest/pytest-style backend tests, Vue 3, Axios, Element Plus, Vite.

---

### File Structure

**Backend**
- Modify: `backend/app/services/auth_service.py`
  - Allow both admin roles to authenticate and seed the initial account as `super_admin`.
- Modify: `backend/app/api/v1/auth_deps.py`
  - Keep `require_admin` for both roles and add `require_super_admin`.
- Modify: `backend/app/db/auth_user_repository.py`
  - Add user listing and role-aware create helpers.
- Modify: `backend/app/api/v1/admin/__init__.py`
  - Register the new admin user router.
- Create: `backend/app/api/v1/admin/users.py`
  - Add `GET /admin/users` and `POST /admin/users/sub-admin`.
- Test: `backend/tests/test_auth_service.py`
  - Cover sub-admin login and super-admin seeding behavior.
- Test: `backend/tests/test_auth_permissions.py`
  - Cover both-role access and super-admin-only endpoint protection.

**Frontend**
- Modify: `frontend/src/services/api.js`
  - Add admin-user management API methods.
- Modify: `frontend/src/App.vue`
  - Show the new menu item only for `super_admin`.
- Modify: `frontend/src/utils/adminNavigation.mjs`
  - Mark the new menu item as part of admin navigation.
- Create: `frontend/src/components/admin/AdminUserManagement.vue`
  - Minimal form + table for sub-admin issuance.

### Task 1: Lock In Backend Role Semantics

**Files:**
- Modify: `backend/tests/test_auth_service.py`
- Modify: `backend/tests/test_auth_permissions.py`
- Modify: `backend/app/services/auth_service.py`
- Modify: `backend/app/api/v1/auth_deps.py`

- [ ] **Step 1: Write the failing auth service tests**

```python
    def test_authenticate_admin_accepts_sub_admin_role(self):
        repo = Mock()
        password_hash = auth_service.hash_password("88888888")
        repo.get_by_username.return_value = {
            "id": "user-2",
            "username": "child-admin",
            "password_hash": password_hash,
            "role": "sub_admin",
            "is_active": True,
        }

        with patch("app.services.auth_service.get_auth_user_repository", return_value=repo):
            user = auth_service.authenticate_admin("child-admin", "88888888")

        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "sub_admin")

    def test_seed_admin_creates_super_admin_role(self):
        repo = Mock()
        repo.get_by_username.return_value = None

        auth_service.seed_admin_user(
            username="admin",
            password="admin-secret",
            repo=repo,
        )

        kwargs = repo.create_user.call_args.kwargs
        self.assertEqual(kwargs["role"], "super_admin")
```

- [ ] **Step 2: Run the auth service tests to verify they fail**

Run: `pytest backend/tests/test_auth_service.py -v`
Expected: FAIL because `authenticate_admin` only accepts `admin` and `seed_admin_user` still writes `admin`.

- [ ] **Step 3: Write the failing permission tests**

```python
    def test_admin_dependency_accepts_sub_admin_token(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(admin=Depends(require_admin)):
            return {"username": admin["username"], "role": admin["role"]}

        token = auth_service.create_access_token(
            subject="user-2",
            username="child-admin",
            role="sub_admin",
        )

        response = TestClient(app).get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"username": "child-admin", "role": "sub_admin"})
```

- [ ] **Step 4: Run the permission tests to verify they fail**

Run: `pytest backend/tests/test_auth_permissions.py -v`
Expected: FAIL because `require_admin` still rejects any role other than `admin`.

- [ ] **Step 5: Write the minimal implementation**

```python
ALLOWED_ADMIN_ROLES = {"super_admin", "sub_admin"}

def authenticate_admin(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_auth_user_repository().get_by_username(username)
    if not user or not user.get("is_active"):
        return None
    if user.get("role") not in ALLOWED_ADMIN_ROLES:
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None
    return user

def seed_admin_user(username: str, password: str, repo=None) -> Dict[str, Any]:
    ...
    return repo.create_user(username=username, password_hash=password_hash, role="super_admin")
```

```python
def require_admin(user: Dict[str, Any] = Depends(require_authenticated_user)) -> Dict[str, Any]:
    if user.get("role") not in {"super_admin", "sub_admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
    return user

def require_super_admin(user: Dict[str, Any] = Depends(require_authenticated_user)) -> Dict[str, Any]:
    if user.get("role") != "super_admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅主管理员可操作")
    return user
```

- [ ] **Step 6: Run both backend auth test files and verify they pass**

Run: `pytest backend/tests/test_auth_service.py backend/tests/test_auth_permissions.py -v`
Expected: PASS.

### Task 2: Add Super-Admin-Only Sub-Admin Management API

**Files:**
- Create: `backend/app/api/v1/admin/users.py`
- Modify: `backend/app/api/v1/admin/__init__.py`
- Modify: `backend/app/db/auth_user_repository.py`
- Modify: `backend/tests/test_auth_permissions.py`

- [ ] **Step 1: Write the failing API tests**

```python
    def test_super_admin_can_create_sub_admin_with_default_password(self):
        app = create_app()
        token = auth_service.create_access_token(
            subject="user-1",
            username="admin",
            role="super_admin",
        )

        with patch("app.api.v1.admin.users.get_auth_user_repository") as mock_get_repo:
            repo = mock_get_repo.return_value
            repo.get_by_username.return_value = None
            repo.create_user.return_value = {
                "id": "user-3",
                "username": "ops01",
                "password_hash": auth_service.hash_password("88888888"),
                "role": "sub_admin",
                "is_active": True,
                "created_at": None,
                "updated_at": None,
            }

            response = TestClient(app).post(
                "/api/v1/admin/users/sub-admin",
                headers={"Authorization": f"Bearer {token}"},
                json={"username": "ops01"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["user"]["role"], "sub_admin")

    def test_sub_admin_cannot_create_sub_admin(self):
        app = create_app()
        token = auth_service.create_access_token(
            subject="user-2",
            username="child-admin",
            role="sub_admin",
        )

        response = TestClient(app).post(
            "/api/v1/admin/users/sub-admin",
            headers={"Authorization": f"Bearer {token}"},
            json={"username": "ops01"},
        )

        self.assertEqual(response.status_code, 403)
```

- [ ] **Step 2: Run the permission tests to verify they fail**

Run: `pytest backend/tests/test_auth_permissions.py -v`
Expected: FAIL because the route does not exist yet.

- [ ] **Step 3: Add repository and router implementation**

```python
class AuthUserRepository(BaseRepository):
    def list_users(self) -> list[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM auth_user ORDER BY created_at DESC, username ASC"
        )
        return [self._normalize(row) for row in rows]
```

```python
router = APIRouter(prefix="/users", tags=["admin-users"])

DEFAULT_SUB_ADMIN_PASSWORD = "88888888"

class CreateSubAdminRequest(BaseModel):
    username: str

@router.get("")
async def list_admin_users(super_admin=Depends(require_super_admin)):
    repo = get_auth_user_repository()
    users = [
        {"id": user["id"], "username": user["username"], "role": user["role"], "is_active": user["is_active"]}
        for user in repo.list_users()
    ]
    return JSONResponse(content={"success": True, "data": {"users": users}})

@router.post("/sub-admin")
async def create_sub_admin(body: CreateSubAdminRequest, super_admin=Depends(require_super_admin)):
    repo = get_auth_user_repository()
    if repo.get_by_username(body.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="账号已存在")
    user = repo.create_user(
        username=body.username,
        password_hash=hash_password(DEFAULT_SUB_ADMIN_PASSWORD),
        role="sub_admin",
    )
    return JSONResponse(content={"success": True, "data": {"user": {"id": user["id"], "username": user["username"], "role": user["role"]}, "default_password": DEFAULT_SUB_ADMIN_PASSWORD}})
```

- [ ] **Step 4: Register the router**

```python
from .users import router as users_router

admin_router.include_router(users_router)
```

- [ ] **Step 5: Run the permission tests and verify they pass**

Run: `pytest backend/tests/test_auth_permissions.py -v`
Expected: PASS, with both role-protection cases green.

### Task 3: Add Minimal Frontend Sub-Admin Issuance Page

**Files:**
- Create: `frontend/src/components/admin/AdminUserManagement.vue`
- Modify: `frontend/src/services/api.js`
- Modify: `frontend/src/utils/adminNavigation.mjs`
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Add frontend API methods**

```javascript
  async listAdminUsers() {
    const response = await api.get('/admin/users')
    return response.data
  },

  async createSubAdmin(username) {
    const response = await api.post('/admin/users/sub-admin', { username })
    return response.data
  },
```

- [ ] **Step 2: Create the management component**

```vue
<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { apiService } from '../../services/api'

const username = ref('')
const loading = ref(false)
const users = ref([])

const loadUsers = async () => {
  const response = await apiService.listAdminUsers()
  users.value = response.data?.users || []
}

const createUser = async () => {
  if (!username.value.trim()) {
    ElMessage.warning('请输入子管理员账号')
    return
  }
  loading.value = true
  try {
    const response = await apiService.createSubAdmin(username.value.trim())
    ElMessage.success(`创建成功，默认密码：${response.data.default_password}`)
    username.value = ''
    await loadUsers()
  } finally {
    loading.value = false
  }
}

onMounted(loadUsers)
</script>
```

- [ ] **Step 3: Wire the page into the admin shell**

```javascript
const isSuperAdmin = computed(() => currentAdmin.value?.role === 'super_admin')
const isAdmin = computed(() => ['super_admin', 'sub_admin'].includes(currentAdmin.value?.role))

const adminNavItems = [
  { key: 'admin-service-tickets', label: '服务记录/工单', icon: 'Tickets' },
  { key: 'admin-data-import', label: '数据导入', icon: 'UploadFilled' },
  { key: 'admin-data-view', label: '数据查看', icon: 'View' },
  { key: 'admin-collections', label: '系统设置', icon: 'Setting', panelKeys: ['admin-collections', 'admin-create', 'admin-config'] },
  { key: 'admin-users', label: '管理员账号', icon: 'UserFilled', superOnly: true },
]

const visibleAdminNavItems = computed(() =>
  isAdmin.value
    ? adminNavItems.filter((item) => !item.superOnly || isSuperAdmin.value)
    : []
)
```

- [ ] **Step 4: Render the component for `admin-users`**

```vue
<div v-else-if="activeMenu === 'admin-users'"><AdminUserManagement /></div>
```

- [ ] **Step 5: Run the frontend build**

Run: `npm run build`
Working directory: `frontend`
Expected: build succeeds with the new management page included.

### Task 4: Verify the End-to-End Change

**Files:**
- Test-only verification against modified files above

- [ ] **Step 1: Run the targeted backend tests**

Run: `pytest backend/tests/test_auth_service.py backend/tests/test_auth_permissions.py -v`
Expected: PASS.

- [ ] **Step 2: Run the frontend build again**

Run: `npm run build`
Working directory: `frontend`
Expected: PASS.

- [ ] **Step 3: Commit once verification is green**

```bash
git add backend/app/services/auth_service.py backend/app/api/v1/auth_deps.py backend/app/db/auth_user_repository.py backend/app/api/v1/admin/__init__.py backend/app/api/v1/admin/users.py backend/tests/test_auth_service.py backend/tests/test_auth_permissions.py frontend/src/services/api.js frontend/src/utils/adminNavigation.mjs frontend/src/App.vue frontend/src/components/admin/AdminUserManagement.vue docs/superpowers/plans/2026-05-28-admin-sub-admin-management.md
git commit -m "feat: add super admin sub admin management"
```
