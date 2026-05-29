const ENTRY_ID_RE = /^[A-Za-z0-9_.@-]{1,120}$/
const ENTRY_STORAGE_KEY = 'rag_entry_identity'
const ADMIN_QUERY_FLAGS = new Set(['1', 'true', 'yes'])

const clean = (value) => String(value || '').trim()
const encodeHeaderText = (value) => encodeURIComponent(clean(value))

const emptyEntry = () => ({
  userId: '',
  nickname: '',
  kb: '',
  senderId: '',
  requesterName: '',
  channel: 'web',
  entryUserId: '',
  entryUserName: '',
  entrySource: '',
  isStoreEntry: false,
  isAdminEntry: false,
})

const pickFirst = (...values) => {
  for (const value of values) {
    const cleaned = clean(value)
    if (cleaned) return cleaned
  }
  return ''
}

const isAdminQuery = (params) => {
  const adminFlag = clean(params.get('admin')).toLowerCase()
  const modeFlag = clean(params.get('mode')).toLowerCase()
  return ADMIN_QUERY_FLAGS.has(adminFlag) || modeFlag === 'admin'
}

const normalizePathname = (pathname = '/') => {
  const normalized = clean(pathname) || '/'
  if (normalized === '/admin' || normalized === '/admin/') return '/admin'
  return normalized.replace(/\/+$/, '') || '/'
}

export const isAdminEntryLocation = (locationLike = globalThis.location) => {
  const pathname = normalizePathname(locationLike?.pathname || '/')
  if (pathname === '/admin') return true
  const params = new URLSearchParams(locationLike?.search || '')
  return isAdminQuery(params)
}

export const parseEntryParams = (search = '') => {
  const params = new URLSearchParams(search || '')
  const hasPlatformEntry = params.has('userId') || params.has('nickName')
  const hasLegacyEntry = params.has('user_id') || params.has('nickname')
  const isAdminEntry = isAdminQuery(params)
  const userId = pickFirst(params.get('userId'), params.get('user_id'))
  const nickname = pickFirst(params.get('nickName'), params.get('nickname'))
  const kb = clean(params.get('kb'))
  const senderId = clean(params.get('sender_id'))
  const requesterName = clean(params.get('requester_name'))
  const rawChannel = clean(params.get('channel')).toLowerCase()
  const channel = rawChannel || (userId ? 'h5' : 'web')
  const entrySource = pickFirst(
    params.get('entry_source'),
    hasPlatformEntry ? 'renruikeji_sso' : '',
    hasLegacyEntry ? 'legacy_query' : '',
  )

  return {
    userId,
    nickname,
    kb,
    senderId,
    requesterName,
    channel,
    entryUserId: userId,
    entryUserName: nickname,
    entrySource,
    isStoreEntry: Boolean(!isAdminEntry && userId && ENTRY_ID_RE.test(userId)),
    isAdminEntry,
  }
}

export const persistEntryIdentity = (
  entry,
  storage = globalThis.sessionStorage,
) => {
  if (!storage?.setItem) return { ...emptyEntry(), ...(entry || {}) }
  const normalized = { ...emptyEntry(), ...(entry || {}) }
  storage.setItem(ENTRY_STORAGE_KEY, JSON.stringify(normalized))
  return normalized
}

export const clearStoredEntryIdentity = (
  storage = globalThis.sessionStorage,
) => {
  storage?.removeItem?.(ENTRY_STORAGE_KEY)
  return emptyEntry()
}

export const loadStoredEntryIdentity = (
  storage = globalThis.sessionStorage,
) => {
  if (!storage?.getItem) return emptyEntry()
  try {
    const raw = storage.getItem(ENTRY_STORAGE_KEY)
    if (!raw) return emptyEntry()
    return { ...emptyEntry(), ...JSON.parse(raw) }
  } catch {
    return emptyEntry()
  }
}

export const getEntryIdentity = (
  locationLike = globalThis.location,
  storage = globalThis.sessionStorage,
) => {
  const parsed = parseEntryParams(locationLike?.search || '')
  if (isAdminEntryLocation(locationLike) || parsed.isAdminEntry) {
    clearStoredEntryIdentity(storage)
    return { ...emptyEntry(), isAdminEntry: true }
  }
  if (parsed.isStoreEntry) return persistEntryIdentity(parsed, storage)
  const stored = loadStoredEntryIdentity(storage)
  return stored.isStoreEntry ? stored : parsed
}

export const bootstrapEntryIdentity = (
  locationLike = globalThis.location,
  storage = globalThis.sessionStorage,
) => {
  const entry = getEntryIdentity(locationLike, storage)
  if (entry.isAdminEntry) return entry
  return persistEntryIdentity(entry, storage)
}

export const buildEntryHeaders = (entry = getEntryIdentity()) => {
  const headers = {}
  const entryUserId = clean(entry?.entryUserId || entry?.userId)
  const entryUserName = clean(entry?.entryUserName || entry?.nickname)
  const entrySource = clean(entry?.entrySource)
  const channel = clean(entry?.channel).toLowerCase() || 'web'
  if (entryUserId && ENTRY_ID_RE.test(entryUserId)) {
    headers['X-Entry-User-Id'] = entryUserId
    if (entryUserName) headers['X-Entry-User-Name'] = encodeHeaderText(entryUserName)
    if (entrySource) headers['X-Entry-Source'] = entrySource

    headers['X-User-Id'] = entryUserId
    if (entryUserName) headers['X-User-Name'] = encodeHeaderText(entryUserName)
    headers['X-Channel'] = channel || 'h5'

    const senderId = clean(entry?.senderId)
    const requesterName = clean(entry?.requesterName)
    if (senderId && ENTRY_ID_RE.test(senderId)) headers['X-Sender-Id'] = senderId
    if (requesterName) headers['X-Requester-Name'] = encodeHeaderText(requesterName)
  }
  return headers
}

export const getDailySessionTitle = (date = new Date()) => {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
  return `${local.toISOString().slice(0, 10)} 每日会话`
}
