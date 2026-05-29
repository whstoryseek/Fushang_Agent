export function fileIdentity(file) {
  if (!file) return ''
  return [file.name || '', file.size || 0, file.lastModified || 0, file.type || ''].join(':')
}

export function mergeSelectedQueryImages(existingFiles = [], selectedFiles = [], maxCount = 6) {
  const merged = []
  const seen = new Set()

  for (const file of [...existingFiles, ...selectedFiles]) {
    if (!file) continue
    const key = fileIdentity(file)
    if (seen.has(key)) continue
    seen.add(key)
    merged.push(file)
    if (merged.length >= maxCount) break
  }

  return merged
}
