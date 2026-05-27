import assert from 'node:assert/strict'
import test from 'node:test'

import { buildListSessionsParams } from '../src/utils/conversationParams.mjs'

test('conversation list params do not override request identity', () => {
  assert.deepEqual(buildListSessionsParams(), {})
  assert.deepEqual(buildListSessionsParams(''), {})
  assert.deepEqual(buildListSessionsParams(' fushang '), { kb_name: 'fushang' })
  assert.equal(Object.hasOwn(buildListSessionsParams('fushang'), 'user_id'), false)
})
