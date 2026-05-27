const ENTRY_ID_RE = /^[A-Za-z0-9_.@-]{1,120}$/

const clean = (value) => String(value || '').trim()
const encodeHeaderText = (value) => encodeURIComponent(clean(value))

export const parseEntryParams = (search = '') => {
  const params = new URLSearchParams(search || '')
  const userId = clean(params.get('user_id'))
  const nickname = clean(params.get('nickname'))
  const kb = clean(params.get('kb'))
  const senderId = clean(params.get('sender_id'))
  const requesterName = clean(params.get('requester_name'))
  const rawChannel = clean(params.get('channel')).toLowerCase()
  const channel = rawChannel || (userId ? 'h5' : 'web')

  return {
    userId,
    nickname,
    kb,
    senderId,
    requesterName,
    channel,
    isStoreEntry: Boolean(userId && ENTRY_ID_RE.test(userId)),
  }
}

export const getEntryIdentity = (locationLike = globalThis.location) =>
  parseEntryParams(locationLike?.search || '')

export const buildEntryHeaders = (entry = getEntryIdentity()) => {
  const headers = {}
  if (entry?.userId && ENTRY_ID_RE.test(entry.userId)) {
    headers['X-User-Id'] = entry.userId
    if (entry.nickname) headers['X-User-Name'] = encodeHeaderText(entry.nickname)
    headers['X-Channel'] = entry.channel || 'h5'
    if (entry.senderId && ENTRY_ID_RE.test(entry.senderId)) headers['X-Sender-Id'] = entry.senderId
    if (entry.requesterName) headers['X-Requester-Name'] = encodeHeaderText(entry.requesterName)
  }
  return headers
}

export const getDailySessionTitle = (date = new Date()) => {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
  return `${local.toISOString().slice(0, 10)} 每日会话`
}
