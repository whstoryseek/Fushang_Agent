import assert from 'node:assert/strict'
import test from 'node:test'

import { mergeSelectedQueryImages } from '../src/utils/queryImages.mjs'

const fakeFile = (name, size, lastModified, type = 'image/png') => ({
  name,
  size,
  lastModified,
  type,
})

test('mergeSelectedQueryImages keeps multiple selected files', () => {
  const files = mergeSelectedQueryImages([], [
    fakeFile('a.png', 10, 1),
    fakeFile('b.jpg', 20, 2, 'image/jpeg'),
    fakeFile('c.webp', 30, 3, 'image/webp'),
  ])

  assert.equal(files.length, 3)
  assert.deepEqual(files.map(file => file.name), ['a.png', 'b.jpg', 'c.webp'])
})

test('mergeSelectedQueryImages deduplicates repeated files', () => {
  const repeated = fakeFile('a.png', 10, 1)
  const files = mergeSelectedQueryImages([repeated], [repeated, fakeFile('b.png', 20, 2)])

  assert.equal(files.length, 2)
  assert.deepEqual(files.map(file => file.name), ['a.png', 'b.png'])
})

test('mergeSelectedQueryImages caps total file count', () => {
  const files = mergeSelectedQueryImages(
    [],
    [
      fakeFile('1.png', 1, 1),
      fakeFile('2.png', 2, 2),
      fakeFile('3.png', 3, 3),
    ],
    2,
  )

  assert.equal(files.length, 2)
  assert.deepEqual(files.map(file => file.name), ['1.png', '2.png'])
})
