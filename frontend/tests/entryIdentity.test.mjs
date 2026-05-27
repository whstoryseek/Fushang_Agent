import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildEntryHeaders,
  getDailySessionTitle,
  parseEntryParams,
} from '../src/utils/entryIdentity.mjs'

test('parses store H5 identity from URL params', () => {
  const result = parseEntryParams('?user_id=store-1&nickname=%E5%BC%A0%E5%BA%97%E9%95%BF&kb=fushang&sender_id=sender-9&requester_name=%E6%9D%8E%E5%BA%97%E9%95%BF')

  assert.deepEqual(result, {
    userId: 'store-1',
    nickname: '张店长',
    kb: 'fushang',
    senderId: 'sender-9',
    requesterName: '李店长',
    channel: 'h5',
    isStoreEntry: true,
  })
})

test('falls back to web channel when URL has no store identity', () => {
  const result = parseEntryParams('')

  assert.equal(result.userId, '')
  assert.equal(result.nickname, '')
  assert.equal(result.kb, '')
  assert.equal(result.senderId, '')
  assert.equal(result.requesterName, '')
  assert.equal(result.channel, 'web')
  assert.equal(result.isStoreEntry, false)
})

test('filters unsupported identity characters before sending headers', () => {
  const entry = parseEntryParams('?user_id=store_1@fushang&nickname=%20%E5%BA%97%E9%95%BF%20&channel=wecom')

  assert.deepEqual(buildEntryHeaders(entry), {
    'X-User-Id': 'store_1@fushang',
    'X-User-Name': '%E5%BA%97%E9%95%BF',
    'X-Channel': 'wecom',
  })

  const withSender = parseEntryParams('?user_id=store-1&sender_id=sender-1&requester_name=%E6%9D%8E%E5%BA%97%E9%95%BF')
  assert.equal(buildEntryHeaders(withSender)['X-Sender-Id'], 'sender-1')
  assert.equal(buildEntryHeaders(withSender)['X-Requester-Name'], '%E6%9D%8E%E5%BA%97%E9%95%BF')

  assert.equal(buildEntryHeaders(parseEntryParams('?user_id=<script>')).hasOwnProperty('X-User-Id'), false)
})

test('builds stable daily session titles for store entry', () => {
  assert.equal(getDailySessionTitle(new Date('2026-05-26T08:30:00+08:00')), '2026-05-26 每日会话')
})
