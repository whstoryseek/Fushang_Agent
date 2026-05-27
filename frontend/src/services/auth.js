import axios from 'axios'
import { buildEntryHeaders } from '../utils/entryIdentity.mjs'

const TOKEN_KEY = 'rag_admin_token'
const USER_KEY = 'rag_admin_user'
const GUEST_KEY = 'rag_guest_id'
export const AUTH_EXPIRED_EVENT = 'rag-auth-expired'

const makeGuestId = () => {
  const raw = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `guest_${raw.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 48)}`
}

export const getGuestId = () => {
  let guestId = localStorage.getItem(GUEST_KEY)
  if (!guestId || !/^guest_[A-Za-z0-9_-]{8,80}$/.test(guestId)) {
    guestId = makeGuestId()
    localStorage.setItem(GUEST_KEY, guestId)
  }
  return guestId
}

export const getAuthToken = () => localStorage.getItem(TOKEN_KEY) || ''

export const getStoredAdmin = () => {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export const setAuthSession = ({ token, user }) => {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export const clearAuthSession = () => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export const attachAuthHeaders = (config = {}) => {
  config.headers = config.headers || {}
  config.headers['X-Guest-Id'] = getGuestId()
  Object.assign(config.headers, buildEntryHeaders())
  const token = getAuthToken()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
}

export const handleAuthError = (error) => {
  if (error?.response?.status === 401 && getAuthToken()) {
    clearAuthSession()
    window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT))
  }
  return Promise.reject(error)
}

axios.interceptors.request.use(attachAuthHeaders)
axios.interceptors.response.use((response) => response, handleAuthError)

export const authService = {
  async login(username, password) {
    const response = await axios.post('/api/v1/auth/login', { username, password })
    const data = response.data?.data || {}
    setAuthSession({ token: data.access_token, user: data.user })
    return data.user
  },

  async me() {
    if (!getAuthToken()) return null
    const response = await axios.get('/api/v1/auth/me')
    const user = response.data?.data || null
    if (user) setAuthSession({ token: getAuthToken(), user })
    return user
  },

  async logout() {
    try {
      if (getAuthToken()) await axios.post('/api/v1/auth/logout')
    } finally {
      clearAuthSession()
    }
  },
}
