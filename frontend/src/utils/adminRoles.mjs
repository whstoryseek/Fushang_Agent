export const isAdminRole = (role) => role === 'super_admin' || role === 'sub_admin'

export const canManageAdminUsers = (role) => role === 'super_admin'
