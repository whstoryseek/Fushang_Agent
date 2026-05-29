import assert from 'node:assert/strict'
import test from 'node:test'

import {
  canManageAdminUsers,
  isAdminRole,
} from '../src/utils/adminRoles.mjs'

test('recognizes both super admin and sub admin as admin roles', () => {
  assert.equal(isAdminRole('super_admin'), true)
  assert.equal(isAdminRole('sub_admin'), true)
  assert.equal(isAdminRole('admin'), false)
  assert.equal(isAdminRole('guest'), false)
})

test('only super admin can manage admin users', () => {
  assert.equal(canManageAdminUsers('super_admin'), true)
  assert.equal(canManageAdminUsers('sub_admin'), false)
  assert.equal(canManageAdminUsers('admin'), false)
  assert.equal(canManageAdminUsers(undefined), false)
})
