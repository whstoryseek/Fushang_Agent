import assert from 'node:assert/strict'
import test from 'node:test'

import { buildAdminConfigRows } from '../src/utils/adminConfig.mjs'

test('builds visible rows from the current admin config response', () => {
  const rows = buildAdminConfigRows({
    milvus: {
      host: 'localhost',
      port: 19530,
      collections: ['fushang', 'demo'],
    },
    postgres: {
      host: 'localhost',
      port: 5433,
      db: 'knowledge_db',
    },
    embedding: {
      model: 'doubao-embedding-text-240715',
      dimension: 2560,
    },
  })

  assert.deepEqual(rows, [
    { label: 'Milvus 地址', value: 'localhost:19530' },
    { label: 'PostgreSQL 数据库', value: 'localhost:5433/knowledge_db' },
    { label: '知识库集合', value: 'fushang, demo' },
    { label: 'Embedding 模型', value: 'doubao-embedding-text-240715' },
    { label: 'Embedding 维度', value: '2560' },
  ])
})

test('keeps legacy config responses displayable', () => {
  const rows = buildAdminConfigRows({
    instance_id: 'adb-prod',
    region_id: 'cn-shanghai',
    namespace: 'knowledge_ns',
    collection: 'fushang',
    embedding_model: 'old-model',
  })

  assert.deepEqual(rows, [
    { label: '实例 ID', value: 'adb-prod' },
    { label: '区域', value: 'cn-shanghai' },
    { label: '命名空间', value: 'knowledge_ns' },
    { label: '当前文档集合', value: 'fushang' },
    { label: 'Embedding 模型', value: 'old-model' },
  ])
})
