// Локально проверяем логику Apps Script без обращения к настоящей таблице.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
context.Utilities = {
  getUuid: () => require('node:crypto').randomUUID(),
  formatDate: date => date.toISOString().slice(0, 19).replace('T', ' '),
};
vm.runInContext(fs.readFileSync('worker/google_sheets/Code.js', 'utf8'), context);
const fields = vm.runInContext('FIELDS', context);
let cells = [];
const sheet = {
  getLastColumn: () => Math.max(0, ...cells.map(row => row.length)),
  getLastRow: () => cells.length,
  setFrozenRows: () => {},
  getRange(row, col, height = 1, width = 1) {
    return {
      getDisplayValues: () => Array.from({length: height}, (_, i) =>
        Array.from({length: width}, (_, j) => String(cells[row - 1 + i]?.[col - 1 + j] ?? ''))),
      setValues(values) {
        values.forEach((valuesRow, i) => valuesRow.forEach((value, j) => {
          cells[row - 1 + i] ??= [];
          cells[row - 1 + i][col - 1 + j] = value;
        }));
      },
      setValue(value) { this.setValues([[value]]); },
      setNumberFormat() { return this; },
    };
  },
};
const execute = request => context.executePlan(sheet, request);
execute({action: 'initialize'});
assert.deepEqual(cells[0], Array.from(fields));
const row = Object.fromEntries(Array.from(fields, name => [name, '']));
Object.assign(row, {benchmark: 'gaia', model: 'm', backend: 'external', feature: 'budget_tracker'});
cells.push(Array.from(fields, name => row[name]));
const request = {action: 'claim', model: 'm', backend: 'external', worker_id: 'alice'};
const claimed = execute(request);
assert.equal(claimed.worker_id, 'alice');
assert.match(claimed.job_id, /^[a-f0-9]{8}$/);
assert.match(claimed.started_at, /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/);
assert.equal(execute({...request, worker_id: 'bob'}), null);
execute({action: 'update', job_id: claimed.job_id, worker_id: 'alice',
  values: {worker_id: 'alice', full_time_sum: '42', max_input_tokens: '1234', tool_calls: '2', status: 'running'}});
assert.equal(cells[1][fields.indexOf('full_time_sum')], '42');
assert.equal(cells[1][fields.indexOf('max_input_tokens')], '1234');
assert.equal(cells[1][fields.indexOf('status')], 'running');
assert.equal(cells[1][fields.indexOf('tool_calls')], '2');
assert.throws(() => execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'bob', status: 'done'}}), /ownership/);
execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'alice', status: 'done', error: ''}});
assert.equal(cells[1][fields.indexOf('status')], 'done');
assert.equal(cells[1][fields.indexOf('feature')], 'budget_tracker');
cells[1][fields.indexOf('status')] = '';
const retry = execute(request);
assert.equal(retry, null);
const otherWorker = execute({...request, worker_id: 'bob'});
assert.equal(otherWorker.job_id, claimed.job_id);
assert.equal(otherWorker.worker_id, 'bob');
cells[1][fields.indexOf('status')] = '';
assert.equal(execute({...request, exclude_job_ids: [claimed.job_id]}), null);
const nextWorker = execute({...request, worker_id: 'charlie'});
assert.equal(nextWorker.job_id, claimed.job_id);
cells[1][fields.indexOf('status')] = '';
cells.push(Array.from(fields, name => row[name]));
const freshJob = execute({...request, exclude_job_ids: [claimed.job_id]});
assert.notEqual(freshJob.job_id, claimed.job_id);
assert.equal(cells[1][fields.indexOf('status')], '');
// Только просроченные running-строки переводятся в interrupted; живые job не меняются.
cells[1][fields.indexOf('status')] = 'running';
cells[1][fields.indexOf('heartbeat_at')] = '2000-01-01 00:00:00';
assert.equal(execute({action: 'mark_stale_jobs_interrupted', stale_after: 300}), 1);
assert.equal(cells[1][fields.indexOf('status')], 'interrupted');
assert.equal(cells[1][fields.indexOf('result')], 'interrupted');
assert.equal(cells[1][fields.indexOf('error')], 'job:interrupted');
assert.equal(cells[2][fields.indexOf('status')], 'running');
cells[0][0] = 'wrong';
assert.throws(() => execute({action: 'initialize'}), /header/);
assert.equal(cells[0][0], 'wrong');
// Настройка не содержит личной таблицы и не перезаписывает существующие свойства.
cells = [];
const properties = new Map();
context.PropertiesService = {getScriptProperties: () => ({
  getProperty: name => properties.get(name) ?? null,
  setProperty: (name, value) => properties.set(name, value),
  deleteProperty: name => properties.delete(name),
})};
assert.throws(() => context.setup(), /Set SPREADSHEET_ID/);
properties.set('SPREADSHEET_ID', 'test-spreadsheet');
properties.set('SHEET_ID', '7');
properties.set('PLAN_GOOGLE_SHEETS_TOKEN', 'existing-secret');
context.LockService = {getScriptLock: () => ({waitLock() {}, releaseLock() {}})};
context.SpreadsheetApp = {
  openById(id) {
    assert.equal(id, 'test-spreadsheet');
    return {getSheetById(id) { assert.equal(id, 7); return sheet; }};
  },
  flush() {},
};
context.setup();
assert.equal(properties.get('PLAN_GOOGLE_SHEETS_TOKEN'), 'existing-secret');
assert.equal(properties.get('SHEET_ID'), '7');
console.log('Sheets plan: header, claim, ownership and preservation checks passed');
