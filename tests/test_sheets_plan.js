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
assert.equal(fields[fields.indexOf('status') + 1], 'score');
assert.equal(fields[fields.indexOf('tool_time_sum') + 1], 'tps');
assert.equal(fields[fields.indexOf('tool_time_sum') + 2], 'avg_tps');
let cells = [];
const writes = [];
const preparations = [];
function makeSheet(getCells = () => cells) {
let filter = null;
const sheet = {
  setName() {},
  getFilter: () => filter,
  getMaxRows: () => 1000,
  deleteRow: number => getCells().splice(number - 1, 1),
  getLastColumn: () => Math.max(0, ...getCells().map(row => row.length)),
  getLastRow: () => getCells().length,
  setFrozenRows: () => preparations.push('freeze'),
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
const beforePreparation = preparations.length;
execute({action: 'update', job_id: claimed.job_id, worker_id: 'alice',
  values: {worker_id: 'alice', full_time_sum: '42', max_input_tokens: '1234', tool_calls: '2', score: 0, status: 'running'}});
assert.equal(writes.length, beforeUpdate + 1);
assert.equal(cells[1][fields.indexOf('score')], '0');
assert.deepEqual(writes.at(-1), [2, 7, 1, fields.length - 6]);
assert.equal(preparations.length, beforePreparation);
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
cells[1][fields.indexOf('status')] = 'running';
const batch = cells.slice(1).map(row => ({job_id: row[fields.indexOf('job_id')], values: {
  worker_id: row[fields.indexOf('worker_id')], status: 'running', tool_calls: '9',
}}));
const beforeBatch = writes.length;
execute({action: 'update_many', updates: batch});
assert.equal(writes.length, beforeBatch + 1);
assert.deepEqual(writes.at(-1), [2, 7, 2, fields.length - 6]);
execute({action: 'update_many', updates: batch});
assert.equal(cells[1][fields.indexOf('tool_calls')], '9');
assert.equal(cells[2][fields.indexOf('tool_calls')], '9');
const rejected = execute({action: 'update_many', updates: [
  {...batch[0], values: {...batch[0].values, tool_calls: '10'}},
  {...batch[1], values: {...batch[1].values, worker_id: 'wrong-owner'}},
]});
assert.equal(rejected.length, 1);
assert.equal(rejected[0].job_id, batch[1].job_id);
assert.equal(rejected[0].error, 'job_changed');
assert.equal(cells[1][fields.indexOf('tool_calls')], '10');
assert.equal(cells[2][fields.indexOf('tool_calls')], '9');
assert.throws(() => execute({action: 'update', ...batch[1], values: {...batch[1].values, worker_id: 'wrong-owner'}}), /ownership/);
assert.equal(fields.includes('heartbeat_at'), false);
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
assert.equal(experimentFields.indexOf('repeat') + 1, experimentFields.indexOf('experiment_id'));
grids.get('Experiments')().push(['gaia', '1, 2', 'm, n', 'budget_tracker,voi', 'FLAG=yes', '', '']);
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
cells[1][fields.indexOf('started_at')] = '2026-10-25 00:00:00';
cells[1][fields.indexOf('job_id')] = 'job1';
cells[1][fields.indexOf('worker_id')] = 'worker1';
cells[2][fields.indexOf('status')] = 'running';
cells[2][fields.indexOf('started_at')] = '2000-01-01 00:00:00';
cells[3][fields.indexOf('status')] = 'error';
cells[3][fields.indexOf('started_at')] = '2026-10-26 00:00:01';
assert.equal(processAt('2026-10-27T12:30:00Z').archived, 0);
assert.equal(processAt('2026-10-28T00:00:00Z').archived, 1);
assert.equal(grids.get('Completed Jobs')().length, 2);
assert.equal(cells.length, 4);
assert.equal(processAt('2026-10-28T23:00:00Z').archived, 0);
assert.equal(processAt('2026-10-29T00:00:00Z').archived, 1);
assert.equal(cells.filter(row => row[fields.indexOf('status')] === 'running').length, 1);
assert.equal(grids.get('Completed Jobs')().length, 3);
// Повторы одной пары sample/model идут подряд; обработанный эксперимент повторно не размножается.
grids.get('Experiments')().push(['gaia', '1,2', 'm,n', 'empty', '', '3', '']);
assert.equal(processAt('2026-10-29T00:01:00Z').created, 12);
const repeatId = grids.get('Experiments')()[2][experimentFields.indexOf('experiment_id')];
const repeated = cells.slice(1).filter(r => r[fields.indexOf('experiment_id')] === repeatId);
assert.equal(repeated.length, 12);
assert.deepEqual(repeated.map(row => [row[1], row[2]]), [
  ['1','m'], ['1','m'], ['1','m'], ['1','n'], ['1','n'], ['1','n'],
  ['2','m'], ['2','m'], ['2','m'], ['2','n'], ['2','n'], ['2','n'],
]);
for (const index of ['1', '2']) for (const model of ['m', 'n']) {
  assert.equal(repeated.filter(r => r[1] === index && r[2] === model).length, 3);
}
assert.equal(processAt('2026-10-29T00:02:00Z').created, 0);
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
assert.deepEqual(reads, [[2, 1, 2, 7]]);
context.LockService = {getScriptLock: () => ({tryLock: () => false, hasLock: () => false})};
context.ContentService = {MimeType: {JSON: 'application/json'}, createTextOutput: text => ({setMimeType: () => text})};
assert.deepEqual(JSON.parse(context.doPost({postData: {contents: JSON.stringify({token: 'existing-secret', action: 'claim'})}})),
  {ok: false, error: 'plan:busy'});
console.log('Sheets: claim, ownership, filters, experiment expansion, daily archive and trigger checks passed');
