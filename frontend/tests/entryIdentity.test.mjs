import assert from 'node:assert/strict'
import test from 'node:test'

import {
  bootstrapEntryIdentity,
  buildEntryHeaders,
  getDailySessionTitle,
  getEntryIdentity,
  parseEntryParams,
} from '../src/utils/entryIdentity.mjs'

const createStorage = () => {
  const store = new Map()
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null
    },
    setItem(key, value) {
      store.set(key, String(value))
    },
    removeItem(key) {
      store.delete(key)
    },
  }
}

test('parses platform entry identity from URL params', () => {
  const result = parseEntryParams('?userId=wangqizhi_l8el&nickName=%E5%86%9C%E8%B5%84%E5%BA%97%E7%8E%8B%E9%BA%92%E9%BA%9F&kb=fushang&sender_id=sender-9&requester_name=%E6%9D%8E%E5%BA%97%E9%95%BF')

  assert.deepEqual(result, {
    userId: 'wangqizhi_l8el',
    nickname: '农资店王麒麟',
    kb: 'fushang',
    senderId: 'sender-9',
    requesterName: '李店长',
    channel: 'h5',
    entryUserId: 'wangqizhi_l8el',
    entryUserName: '农资店王麒麟',
    entrySource: 'renruikeji_sso',
    isStoreEntry: true,
    isAdminEntry: false,
  })
})

test('falls back to web channel when URL has no store identity', () => {
  const result = parseEntryParams('')

  assert.equal(result.userId, '')
  assert.equal(result.nickname, '')
  assert.equal(result.kb, '')
  assert.equal(result.senderId, '')
  assert.equal(result.requesterName, '')
  assert.equal(result.entryUserId, '')
  assert.equal(result.entryUserName, '')
  assert.equal(result.entrySource, '')
  assert.equal(result.channel, 'web')
  assert.equal(result.isStoreEntry, false)
  assert.equal(result.isAdminEntry, false)
})

test('keeps legacy params compatible and sends entry headers', () => {
  const entry = parseEntryParams('?user_id=store_1@fushang&nickname=%20%E5%BA%97%E9%95%BF%20&channel=wecom')

  assert.deepEqual(buildEntryHeaders(entry), {
    'X-Entry-User-Id': 'store_1@fushang',
    'X-Entry-User-Name': '%E5%BA%97%E9%95%BF',
    'X-Entry-Source': 'legacy_query',
    'X-User-Id': 'store_1@fushang',
    'X-User-Name': '%E5%BA%97%E9%95%BF',
    'X-Channel': 'wecom',
  })

  const withSender = parseEntryParams('?userId=wangqizhi_l8el&nickName=%E5%86%9C%E8%B5%84%E5%BA%97%E7%8E%8B%E9%BA%92%E9%BA%9F&sender_id=sender-1&requester_name=%E6%9D%8E%E5%BA%97%E9%95%BF')
  assert.equal(buildEntryHeaders(withSender)['X-Sender-Id'], 'sender-1')
  assert.equal(buildEntryHeaders(withSender)['X-Requester-Name'], '%E6%9D%8E%E5%BA%97%E9%95%BF')
  assert.equal(buildEntryHeaders(withSender)['X-Entry-User-Id'], 'wangqizhi_l8el')
  assert.equal(buildEntryHeaders(withSender)['X-Entry-Source'], 'renruikeji_sso')

  assert.equal(Object.prototype.hasOwnProperty.call(buildEntryHeaders(parseEntryParams('?userId=<script>')), 'X-Entry-User-Id'), false)
  assert.equal(Object.prototype.hasOwnProperty.call(buildEntryHeaders(parseEntryParams('?userId=<script>')), 'X-User-Id'), false)
})

test('builds stable daily session titles for store entry', () => {
  assert.equal(getDailySessionTitle(new Date('2026-05-26T08:30:00+08:00')), '2026-05-26 每日会话')
})

test('fixed admin pathname ignores cached store identity', () => {
  const storage = createStorage()

  storage.setItem('rag_entry_identity', JSON.stringify({
    userId: 'wangqizhi_l8el',
    nickname: '农资店王麒麟',
    kb: 'fushang',
    senderId: '',
    requesterName: '',
    channel: 'h5',
    entryUserId: 'wangqizhi_l8el',
    entryUserName: '农资店王麒麟',
    entrySource: 'renruikeji_sso',
    isStoreEntry: true,
    isAdminEntry: false,
  }))

  const entry = getEntryIdentity({ pathname: '/admin', search: '' }, storage)

  assert.equal(entry.isAdminEntry, true)
  assert.equal(entry.isStoreEntry, false)
  assert.equal(entry.userId, '')
  assert.equal(entry.nickname, '')
  assert.equal(storage.getItem('rag_entry_identity'), null)
})

test('admin query parameter enters management mode and clears cache on bootstrap', () => {
  const storage = createStorage()

  storage.setItem('rag_entry_identity', JSON.stringify({
    userId: 'store_1@fushang',
    nickname: '店长',
    kb: 'fushang',
    senderId: '',
    requesterName: '',
    channel: 'h5',
    entryUserId: 'store_1@fushang',
    entryUserName: '店长',
    entrySource: 'legacy_query',
    isStoreEntry: true,
    isAdminEntry: false,
  }))

  const entry = bootstrapEntryIdentity({ pathname: '/', search: '?admin=1' }, storage)

  assert.equal(entry.isAdminEntry, true)
  assert.equal(entry.isStoreEntry, false)
  assert.equal(storage.getItem('rag_entry_identity'), null)
  assert.deepEqual(buildEntryHeaders(entry), {})
})
