const missingText = '未配置'

const hasValue = (value) => value !== undefined && value !== null && value !== ''

const endpoint = ({ host, port } = {}) => {
  if (hasValue(host) && hasValue(port)) return `${host}:${port}`
  if (hasValue(host)) return String(host)
  if (hasValue(port)) return String(port)
  return missingText
}

const dbEndpoint = ({ host, port, db } = {}) => {
  const base = endpoint({ host, port })
  return hasValue(db) ? `${base}/${db}` : base
}

const valueOrMissing = (value) => hasValue(value) ? String(value) : missingText

export const buildAdminConfigRows = (config = {}) => {
  if (config.milvus || config.postgres || config.embedding) {
    const collections = Array.isArray(config.milvus?.collections)
      ? config.milvus.collections
      : []

    return [
      { label: 'Milvus 地址', value: endpoint(config.milvus) },
      { label: 'PostgreSQL 数据库', value: dbEndpoint(config.postgres) },
      { label: '知识库集合', value: collections.length ? collections.join(', ') : '暂无集合' },
      { label: 'Embedding 模型', value: valueOrMissing(config.embedding?.model) },
      { label: 'Embedding 维度', value: valueOrMissing(config.embedding?.dimension) },
    ]
  }

  if (
    hasValue(config.instance_id) ||
    hasValue(config.region_id) ||
    hasValue(config.namespace) ||
    hasValue(config.collection) ||
    hasValue(config.embedding_model)
  ) {
    return [
      { label: '实例 ID', value: valueOrMissing(config.instance_id) },
      { label: '区域', value: valueOrMissing(config.region_id) },
      { label: '命名空间', value: valueOrMissing(config.namespace) },
      { label: '当前文档集合', value: valueOrMissing(config.collection) },
      { label: 'Embedding 模型', value: valueOrMissing(config.embedding_model) },
    ]
  }

  return []
}
