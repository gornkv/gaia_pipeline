// В Script Properties задайте SPREADSHEET_ID, SHEET_ID и PLAN_GOOGLE_SHEETS_TOKEN.
// Все worker должны обращаться к одному развёртыванию этого проекта.
const FIELDS = [
  'benchmark', 'index', 'model', 'feature', 'feature_envs', 'experiment_id',
  'worker_id', 'job_id', 'status', 'score', 'result', 'error', 'max_input_tokens', 'tool_calls',
  'tool_timeouts', 'started_at', 'full_time_sum', 'tool_time_sum',
  'tps', 'avg_tps', 'avg_job_time_limit', 'avg_tool_count', 'avg_tool_time', 'jobs_count',
];
const OUTPUT_FIELDS = FIELDS.slice(6);
const EXPERIMENT_FIELDS = ['benchmark', 'samples', 'models', 'features', 'feature_envs', 'repeat', 'experiment_id'];

function initializeSheet(sheet, fields) {
  const width = Math.max(sheet.getLastColumn(), fields.length);
  const header = sheet.getRange(1, 1, 1, width).getDisplayValues()[0];
  if (header.every(value => value === '')) {
    sheet.getRange(1, 1, 1, fields.length).setValues([fields]);
  } else if (fields.some((field, i) => header[i] !== field) || header.slice(fields.length).some(Boolean)) {
    throw new Error('invalid plan header');
  }
  sheet.setFrozenRows(1);
  if (!sheet.getFilter()) sheet.getRange(1, 1, sheet.getMaxRows(), fields.length).createFilter();
  return sheet;
}

function planSheets(book, config) {
  const jobs = book.getSheetByName('Jobs') || book.getSheetById(Number(config.getProperty('SHEET_ID') || '0')) || book.insertSheet('Jobs');
  jobs.setName('Jobs');
  return {
    jobs: initializeSheet(jobs, FIELDS),
    completed: initializeSheet(book.getSheetByName('Completed Jobs') || book.insertSheet('Completed Jobs'), FIELDS),
    experiments: initializeSheet(book.getSheetByName('Experiments') || book.insertSheet('Experiments'), EXPERIMENT_FIELDS),
  };
}

function appendRows(sheet, rows) {
  if (!rows.length) return;
  const start = sheet.getLastRow() + 1;
  const missing = start + rows.length - 1 - sheet.getMaxRows();
  if (missing > 0) sheet.insertRowsAfter(sheet.getMaxRows(), missing);
  sheet.getRange(start, 1, rows.length, rows[0].length).setNumberFormat('@').setValues(rows);
}

function readRows(sheet, fields) {
  const count = sheet.getLastRow() - 1;
  return count > 0 ? sheet.getRange(2, 1, count, fields.length).getDisplayValues() : [];
}

function installTrigger() {
  if (!ScriptApp.getProjectTriggers().some(trigger => trigger.getHandlerFunction() === 'maintainPlan')) {
    ScriptApp.newTrigger('maintainPlan').timeBased().everyMinutes(15).create();
  }
}

function processExperimentsAndArchive(book, config, now = new Date()) {
  const experiments = book.getSheetByName('Experiments');
  let created = 0;
  const list = value => value.split(',').map(item => item.trim()).filter(Boolean);
  readRows(experiments, EXPERIMENT_FIELDS).forEach((cells, i) => {
    const [benchmark, samples, models, features, featureEnvs, repeat, existingId] = cells;
    if (existingId || !benchmark || !samples || !models) return;
    const id = Utilities.getUuid().replace(/-/g, '').slice(0, 8);
    const rows = list(samples).flatMap(index => list(models).flatMap(model => Array.from({length: Number(repeat || 1)}, () => {
      const values = {benchmark, index, model, feature: features, feature_envs: featureEnvs, experiment_id: id};
      return FIELDS.map(field => values[field] || '');
    })));
    appendRows(book.getSheetByName('Jobs'), rows);
    experiments.getRange(i + 2, EXPERIMENT_FIELDS.indexOf('experiment_id') + 1).setNumberFormat('@').setValue(id);
    created += rows.length;
  });

  const day = Utilities.formatDate(now, 'UTC', 'yyyy-MM-dd');
  let archived = 0;
  if (config.getProperty('LAST_ARCHIVE_DAY') !== day) {
    const jobs = book.getSheetByName('Jobs');
    const completed = book.getSheetByName('Completed Jobs');
    const cutoff = Date.parse(day + 'T00:00:00Z') - 2 * 86400000;
    const rows = readRows(jobs, FIELDS);
    const status = FIELDS.indexOf('status'), started = FIELDS.indexOf('started_at');
    const archive = rows.map((cells, i) => ({cells, number: i + 2})).filter(({cells}) =>
      ['done', 'error', 'interrupted'].includes(cells[status]) && Date.parse(cells[started].replace(' ', 'T') + 'Z') <= cutoff);
    // Если копирование удалось, а удаление прервалось, следующий запуск не дублирует архив.
    const key = cells => cells[FIELDS.indexOf('worker_id')] + '/' + cells[FIELDS.indexOf('job_id')];
    const existing = new Set(readRows(completed, FIELDS).map(key));
    appendRows(completed, archive.filter(({cells}) => !existing.has(key(cells))).map(({cells}) => cells));
    SpreadsheetApp.flush();
    archive.reverse().forEach(({number}) => jobs.deleteRow(number));
    archived = archive.length;
    config.setProperty('LAST_ARCHIVE_DAY', day);
  }
  return {created, archived};
}

// Один обработчик и один lock для генерации jobs, архивирования и запросов worker.
function maintainPlan() {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const config = PropertiesService.getScriptProperties();
    const result = processExperimentsAndArchive(SpreadsheetApp.openById(config.getProperty('SPREADSHEET_ID')), config);
    if (result.created || result.archived) SpreadsheetApp.flush();
    return result;
  } finally {
    lock.releaseLock();
  }
}

// Запустите один раз из редактора: Google запросит разрешение работать с таблицей.
function setup() {
  const config = PropertiesService.getScriptProperties();
  // Привязка к таблице хранится в свойствах конкретного проекта, а не в публикуемом коде.
  if (!config.getProperty('SPREADSHEET_ID')) throw new Error('Set SPREADSHEET_ID in Script Properties');
  if (!config.getProperty('PLAN_GOOGLE_SHEETS_TOKEN')) {
    // При переименовании сохраняем уже выданный секрет.
    config.setProperty('PLAN_GOOGLE_SHEETS_TOKEN', config.getProperty('PLAN_TOKEN') || Utilities.getUuid() + Utilities.getUuid());
    config.deleteProperty('PLAN_TOKEN');
  }
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    planSheets(SpreadsheetApp.openById(config.getProperty('SPREADSHEET_ID')), config);
    installTrigger();
    SpreadsheetApp.flush();
  } finally {
    lock.releaseLock();
  }
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  let result;
  try {
    const request = JSON.parse(e.postData.contents);
    const config = PropertiesService.getScriptProperties();
    const token = config.getProperty('PLAN_GOOGLE_SHEETS_TOKEN') || config.getProperty('PLAN_TOKEN');
    if (!token || request.token !== token) throw new Error('unauthorized');
    if (!lock.tryLock(20000)) throw new Error('plan:busy');
    const book = SpreadsheetApp.openById(config.getProperty('SPREADSHEET_ID'));
    const sheet = book.getSheetByName('Jobs') || book.getSheetById(Number(config.getProperty('SHEET_ID') || '0'));
    if (!sheet) throw new Error('sheet not found');
    result = {ok: true, result: executePlan(sheet, request)};
    SpreadsheetApp.flush();
  } catch (error) {
    result = {ok: false, error: error.message};
  } finally {
    if (lock.hasLock()) lock.releaseLock();
  }
  return ContentService.createTextOutput(JSON.stringify(result))
    .setMimeType(ContentService.MimeType.JSON);
}

function executePlan(sheet, request) {
  // Пустая шапка создаётся автоматически; непустую ошибочную шапку не затираем.
  const lastRow = sheet.getLastRow();
  if (request.action === 'initialize' || lastRow === 0) initializeSheet(sheet, FIELDS);
  if (request.action === 'initialize') return null;
  const count = Math.max(0, lastRow - 1);
  const rows = count ? sheet.getRange(2, 1, count, FIELDS.length).getDisplayValues() : [];
  const ids = new Set();
  const jobs = [];
  rows.forEach((cells, i) => {
    if (cells.every(value => value === '')) return;
    const row = Object.fromEntries(FIELDS.map((field, j) => [field, cells[j]]));
    const key = `${row.worker_id}/${row.job_id}`;
    if (row.job_id && (!/^[a-zA-Z0-9]+(?:-[0-9]+)?$/.test(row.job_id) || ids.has(key))) {
      throw new Error('worker_id and job_id must be unique');
    }
    if (row.job_id) ids.add(key);
    jobs.push({row, number: i + 2});
  });
  // Пишем только изменяемые служебные ячейки: настройки эксперимента людей сохраняются.
  function update(changes) {
    const groups = [];
    changes.sort((a, b) => a.job.number - b.job.number).forEach(({job, values}) => {
      const updated = {...job.row, ...values};
      const cells = OUTPUT_FIELDS.map(field => String(updated[field]));
      const group = groups[groups.length - 1];
      if (group && group.number + group.rows.length === job.number) group.rows.push(cells);
      else groups.push({number: job.number, rows: [cells]});
    });
    groups.forEach(({number, rows}) => sheet.getRange(number, 7, rows.length, OUTPUT_FIELDS.length).setValues(rows));
  }
  const now = Utilities.formatDate(new Date(), 'UTC', 'yyyy-MM-dd HH:mm:ss');
  if (request.action === 'claim') {
    const excluded = new Set(request.exclude_job_ids);
    const claimed = jobs.find(({row}) => row.status === 'running' && row.model === request.model
      && row.worker_id === request.worker_id && !excluded.has(row.job_id));
    if (claimed) return claimed.row;
    const job = jobs.find(({row}) => !row.status && row.model === request.model
      && row.worker_id !== request.worker_id && !excluded.has(row.job_id));
    if (!job) return null;
    let id = job.row.job_id;
    if (!id) {
      do { id = Utilities.getUuid().replace(/-/g, '').slice(0, 8); }
      while (jobs.some(({row}) => row.job_id === id));
    }
    const values = Object.fromEntries(OUTPUT_FIELDS.map(field => [field, '']));
    Object.assign(values, {job_id: id, status: 'running', worker_id: request.worker_id, started_at: now});
    update([{job, values}]);
    return {...job.row, ...values};
  }
  if (!['update', 'update_many'].includes(request.action)) throw new Error('unknown action');
  const byId = new Map(jobs.map(job => [`${job.row.worker_id}/${job.row.job_id}`, job]));
  const updates = request.action === 'update_many' ? request.updates : [request];
  const rejected = [];
  const changes = updates.flatMap(item => {
    const owner = item.worker_id || item.values.worker_id;
    const job = byId.get(`${owner}/${item.job_id}`);
    if (!job || job.row.status !== 'running' && job.row.status !== item.values.status || job.row.worker_id !== item.values.worker_id) {
      if (request.action === 'update') throw new Error('job ownership changed');
      rejected.push({job_id: item.job_id, error: 'job_changed'});
      return [];
    }
    if (Object.keys(item.values).some(field => !OUTPUT_FIELDS.includes(field))) throw new Error('invalid output field');
    return [{job, values: item.values}];
  });
  update(changes);
  return request.action === 'update_many' ? rejected : null;
}
