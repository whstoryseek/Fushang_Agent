const adminPanelKeys = new Set(['admin-collections', 'admin-create', 'admin-config'])

export const isAdminPanelMenu = (key) => adminPanelKeys.has(key)

export const isAdminNavItemActive = (item, activeKey) => {
  if (!item || !activeKey) return false
  return item.key === activeKey || Boolean(item.panelKeys?.includes(activeKey))
}
