export const buildListSessionsParams = (kbName = null) => {
  const params = {}
  const normalizedKbName = String(kbName || '').trim()
  if (normalizedKbName) params.kb_name = normalizedKbName
  return params
}
