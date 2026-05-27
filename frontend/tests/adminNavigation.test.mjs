import assert from 'node:assert/strict'
import test from 'node:test'

import {
  isAdminNavItemActive,
  isAdminPanelMenu,
} from '../src/utils/adminNavigation.mjs'

test('shared admin panel is limited to settings-related menus', () => {
  for (const key of ['admin-collections', 'admin-create', 'admin-config']) {
    assert.equal(isAdminPanelMenu(key), true)
  }

  for (const key of ['admin-data-import', 'admin-data-view', 'admin-service-tickets', 'chat', 'user-history']) {
    assert.equal(isAdminPanelMenu(key), false)
  }
})

test('settings nav item is not active for every admin page', () => {
  const settingsItem = {
    key: 'admin-collections',
    panelKeys: ['admin-collections', 'admin-create', 'admin-config'],
  }

  assert.equal(isAdminNavItemActive(settingsItem, 'admin-collections'), true)
  assert.equal(isAdminNavItemActive(settingsItem, 'admin-config'), true)
  assert.equal(isAdminNavItemActive(settingsItem, 'admin-service-tickets'), false)
})
