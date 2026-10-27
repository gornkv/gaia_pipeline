// Локально проверяем логику Apps Script без обращения к настоящей таблице.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
context.Utilities = {
  getUuid: () => require('node:crypto').randomUUID(),
  formatDate: (date, zone, format) => date.toISOString().slice(0, format === 'yyyy-MM-dd' ? 10 : 19).replace('T', ' '),
};
vm.runInContext(fs.readFileSync('worker/google_sheets/Code.js', 'utf8'), context);
const fields = vm.runInContext('FIELDS', context);
let cells = [];
const writes = [];
function makeSheet(getCells = () => cells) {
let filter = null;
const sheet = {
  setName() {},
  getFilter: () => filter,
  getMaxRows: () => 1000,
  deleteRow: number => getCells().splice(number - 1, 1),
  getLastColumn: () => Math.max(0, ...getCells().map(row => row.length)),
  getLastRow: () => getCells().length,
  setFrozenRows: () => {},
  getRange(row, col, height = 1, width = 1) {
    return {
      getDisplayValues: () => Array.from({length: height}, (_, i) =>
        Array.from({length: width}, (_, j) => String(getCells()[row - 1 + i]?.[col - 1 + j] ?? ''))),
      setValues(values) {
        writes.push([row, col, height, width]);
        values.forEach((valuesRow, i) => valuesRow.forEach((value, j) => {
          getCells()[row - 1 + i] ??= [];
          getCells()[row - 1 + i][col - 1 + j] = value;
        }));
      },
      setValue(value) { this.setValues([[value]]); },
      setNumberFormat() { return this; },
      createFilter() { filter = {width}; },
    };
  },
};
return sheet;
}
const sheet = makeSheet();
const execute = request => context.executePlan(sheet, request);
execute({action: 'initialize'});
assert.deepEqual(cells[0], Array.from(fields));
const row = Object.fromEntries(Array.from(fields, name => [name, '']));
Object.assign(row, {benchmark: 'gaia', index: '1', model: 'm', feature: 'budget_tracker', experiment_id: 'experiment1'});
cells.push(Array.from(fields, name => row[name]));
const request = {action: 'claim', model: 'm', worker_id: 'alice', exclude_job_ids: []};
const claimed = execute(request);
assert.equal(claimed.worker_id, 'alice');
assert.equal(claimed.experiment_id, 'experiment1');
assert.equal(sheet.getFilter().width, fields.length);
assert.match(claimed.job_id, /^[a-f0-9]{8}$/);
assert.match(claimed.started_at, /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/);
assert.equal(execute(request).job_id, claimed.job_id);
assert.equal(execute({...request, exclude_job_ids: [claimed.job_id]}), null);
assert.equal(execute({...request, worker_id: 'bob'}), null);
const beforeUpdate = writes.length;
execute({action: 'update', job_id: claimed.job_id, worker_id: 'alice',
  values: {worker_id: 'alice', full_time_sum: '42', max_input_tokens: '1234', tool_calls: '2', status: 'running'}});
assert.equal(writes.length, beforeUpdate + 1);
assert.deepEqual(writes.at(-1), [2, 7, 1, fields.length - 6]);
assert.equal(cells[1][fields.indexOf('full_time_sum')], '42');
assert.equal(cells[1][fields.indexOf('max_input_tokens')], '1234');
assert.equal(cells[1][fields.indexOf('status')], 'running');
assert.equal(cells[1][fields.indexOf('tool_calls')], '2');
assert.throws(() => execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'bob', status: 'done'}}), /ownership/);
execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'alice', status: 'done', error: ''}});
execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'alice', status: 'done', error: ''}});
assert.throws(() => execute({action: 'update', job_id: claimed.job_id, values: {worker_id: 'alice', status: 'running'}}), /ownership/);
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
const sheets = new Map([['Jobs', sheet]]);
const grids = new Map([['Jobs', () => cells]]);
const book = {
  getSheetByName: name => sheets.get(name),
  getSheetById(id) { assert.equal(id, 7); return sheet; },
  insertSheet(name) {
    const rows = [];
    const sheet = makeSheet(() => rows);
    sheets.set(name, sheet);
    grids.set(name, () => rows);
    return sheet;
  },
};
const triggers = [];
context.ScriptApp = {
  getProjectTriggers: () => triggers,
  newTrigger(name) {
    return {timeBased() {return this;}, everyMinutes(n) {assert.equal(n, 15); return this;},
      create() {triggers.push({getHandlerFunction: () => name});}};
  },
};
context.SpreadsheetApp = {
  openById(id) {
    assert.equal(id, 'test-spreadsheet');
    return book;
  },
  flush() {},
};
context.setup();
assert.equal(properties.get('PLAN_GOOGLE_SHEETS_TOKEN'), 'existing-secret');
assert.equal(properties.get('SHEET_ID'), '7');
context.setup();
assert.equal(triggers.length, 1);
const experimentFields = Array.from(vm.runInContext('EXPERIMENT_FIELDS', context));
grids.get('Experiments')().push(['gaia', '1, 2', 'm, n', 'budget_tracker,voi', 'FLAG=yes', '']);
function processAt(date) {
  return context.processExperimentsAndArchive(book, context.PropertiesService.getScriptProperties(), new Date(date));
}
assert.deepEqual({...processAt('2026-10-27T12:00:00Z')}, {created: 4, archived: 0});
assert.equal(cells.length, 5);
const id = grids.get('Experiments')()[1][experimentFields.indexOf('experiment_id')];
assert.match(id, /^[a-f0-9]{8}$/);
for (const job of cells.slice(1)) {
  assert.equal(job[fields.indexOf('experiment_id')], id);
  assert.equal(job[fields.indexOf('feature')], 'budget_tracker,voi');
}
assert.deepEqual(cells.slice(1).map(row => [row[1], row[2]]), [['1','m'],['1','n'],['2','m'],['2','n']]);
assert.equal(processAt('2026-10-27T12:15:00Z').created, 0);
cells[1][fields.indexOf('status')] = 'done';
cells[1][fields.indexOf('heartbeat_at')] = '2026-10-25 00:00:00';
cells[1][fields.indexOf('job_id')] = 'job1';
cells[1][fields.indexOf('worker_id')] = 'worker1';
cells[2][fields.indexOf('status')] = 'running';
cells[2][fields.indexOf('heartbeat_at')] = '2000-01-01 00:00:00';
cells[3][fields.indexOf('status')] = 'error';
cells[3][fields.indexOf('heartbeat_at')] = '2026-10-26 00:00:01';
assert.equal(processAt('2026-10-27T12:30:00Z').archived, 0);
assert.equal(processAt('2026-10-28T00:00:00Z').archived, 1);
assert.equal(grids.get('Completed Jobs')().length, 2);
assert.equal(cells.length, 4);
assert.equal(processAt('2026-10-28T23:00:00Z').archived, 0);
assert.equal(processAt('2026-10-29T00:00:00Z').archived, 1);
assert.equal(cells.filter(row => row[fields.indexOf('status')] === 'running').length, 1);
assert.equal(grids.get('Completed Jobs')().length, 3);
properties.set('LAST_ARCHIVE_DAY', context.Utilities.formatDate(new Date(), 'UTC', 'yyyy-MM-dd'));
const reads = [];
context.SpreadsheetApp = {
  openById: () => ({getSheetByName(name) {
    assert.equal(name, 'Experiments');
    return {
      getLastRow: () => grids.get(name)().length,
      getRange(row, col, height, width) {
        reads.push([row, col, height, width]);
        return {getDisplayValues: () => grids.get(name)().slice(1)};
      },
    };
  }}),
  flush() {throw new Error('Idle run must not flush or prepare sheets');},
};
assert.deepEqual({...context.maintainPlan()}, {created: 0, archived: 0});
assert.deepEqual(reads, [[2, 1, 1, 6]]);
context.LockService = {getScriptLock: () => ({tryLock: () => false, hasLock: () => false})};
context.ContentService = {MimeType: {JSON: 'application/json'}, createTextOutput: text => ({setMimeType: () => text})};
assert.deepEqual(JSON.parse(context.doPost({postData: {contents: JSON.stringify({token: 'existing-secret', action: 'claim'})}})),
  {ok: false, error: 'plan:busy'});
console.log('Sheets: claim, ownership, filters, experiment expansion, daily archive and trigger checks passed');
