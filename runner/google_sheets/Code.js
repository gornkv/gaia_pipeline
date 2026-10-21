// В Script Properties задайте SPREADSHEET_ID, SHEET_ID и PLAN_GOOGLE_SHEETS_TOKEN.
// Все worker должны обращаться к одному развёртыванию этого проекта.
const FIELDS = [
  'benchmark', 'task', 'split', 'sample_no', 'backend', 'model', 'feature', 'feature_envs',
  'worker_id', 'job_id', 'status', 'result', 'error', 'max_input_tokens', 'tool_calls',
  'tool_timeouts', 'started_at', 'heartbeat_at', 'full_time_sum', 'tool_time_sum',
  'avg_job_time_limit', 'avg_tps', 'avg_tool_count', 'avg_tool_time', 'jobs_count',
];
const OUTPUT_FIELDS = FIELDS.slice(8);

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
    const sheet = SpreadsheetApp.openById(config.getProperty('SPREADSHEET_ID'))
      .getSheetById(Number(config.getProperty('SHEET_ID') || '0'));
    if (!sheet) throw new Error('sheet not found');
    executePlan(sheet, {action: 'initialize'});
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
    lock.waitLock(20000);
    const book = SpreadsheetApp.openById(config.getProperty('SPREADSHEET_ID'));
    const sheet = book.getSheetById(Number(config.getProperty('SHEET_ID') || '0'));
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
  const width = Math.max(sheet.getLastColumn(), FIELDS.length);
  const header = sheet.getRange(1, 1, 1, width).getDisplayValues()[0];
  if (header.every(value => value === '')) {
    sheet.getRange(1, 1, 1, FIELDS.length).setValues([FIELDS]);
    sheet.setFrozenRows(1);
  } else if (FIELDS.some((field, i) => header[i] !== field)
      || header.slice(FIELDS.length).some(Boolean)) {
    throw new Error('invalid plan header');
  }
  if (request.action === 'initialize') return null;
  const count = Math.max(0, sheet.getLastRow() - 1);
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
  function update(job, values) {
    for (const [field, value] of Object.entries(values)) {
      if (!OUTPUT_FIELDS.includes(field)) throw new Error('invalid output field');
      const cell = sheet.getRange(job.number, FIELDS.indexOf(field) + 1);
      if (['job_id', 'started_at', 'heartbeat_at'].includes(field)) cell.setNumberFormat('@');
      cell.setValue(String(value));
    }
  }
  const now = Utilities.formatDate(new Date(), 'UTC', 'yyyy-MM-dd HH:mm:ss');
  if (request.action === 'claim') {
    const job = jobs.find(({row}) => !row.status && row.model === request.model && row.backend === request.backend);
    if (!job) return null;
    let id = job.row.job_id;
    if (!id) {
      do { id = Utilities.getUuid().replace(/-/g, '').slice(0, 8); }
      while (jobs.some(({row}) => row.job_id === id));
    } else if (job.row.worker_id === request.worker_id) {
      const match = id.match(/^(.*)-(\d+)$/);
      id = match ? `${match[1]}-${Number(match[2]) + 1}` : `${id}-2`;
      while (ids.has(`${request.worker_id}/${id}`)) {
        const match = id.match(/^(.*)-(\d+)$/);
        id = `${match[1]}-${Number(match[2]) + 1}`;
      }
    }
    const values = Object.fromEntries(OUTPUT_FIELDS.map(field => [field, '']));
    Object.assign(values, {job_id: id, status: 'running', worker_id: request.worker_id, started_at: now, heartbeat_at: now});
    update(job, values);
    return {...job.row, ...values};
  }
  if (request.action === 'recover_orphans') {
    if (!(request.stale_after >= 3)) throw new Error('invalid stale_after');
    let recovered = 0;
    for (const job of jobs) {
      const timestamp = job.row.heartbeat_at || job.row.started_at;
      const last = Date.parse(timestamp.includes('T') ? timestamp : timestamp.replace(' ', 'T') + 'Z');
      if (job.row.status === 'running' && (!Number.isFinite(last) || Date.now() - last > request.stale_after * 1000)) {
        update(job, {status: 'interrupted', result: 'interrupted', error: 'job:interrupted'});
        recovered++;
      }
    }
    return recovered;
  }
  const owner = request.worker_id || request.values?.worker_id;
  const job = jobs.find(({row}) => row.job_id === request.job_id && row.worker_id === owner);
  if (!job) throw new Error('job ownership changed or job disappeared from plan');
  if (request.action === 'heartbeat') {
    if (job.row.status === 'running' && job.row.worker_id === request.worker_id) {
      const values = {heartbeat_at: now};
      // Heartbeat меняет только прогресс, не статус и не итоговую статистику worker.
      for (const field of ['full_time_sum', 'max_input_tokens']) {
        if (request.values?.[field] !== undefined) values[field] = request.values[field];
      }
      update(job, values);
    }
  } else if (request.action === 'finish') {
    if (job.row.status !== 'running' || job.row.worker_id !== request.values.worker_id) throw new Error('job ownership changed');
    update(job, request.values);
  } else {
    throw new Error('unknown action');
  }
  return null;
}
