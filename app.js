/**
 * app.js (fixed)
 *
 * Fixes required by homework:
 * 1) CSV comma/quote escape bug: replaces naive split(',') with a robust CSV parser
 *    that correctly handles quoted fields with commas (e.g., Titanic "Name").
 * 2) Evaluation table not showing: fixes tfjs-vis callbacks misuse and shape issues
 *    (predictions are [N,1] => flatten); also ensures metric DOM elements exist.
 * 3) Adds an in-app code summary (LLM-style) to understand logic.
 * 4) Adds a Sigmoid gate layer to interpret feature importance (learnable per-feature gate).
 *
 * NOTE: This file assumes index.html contains elements with these ids:
 * train-file, test-file, data-status, inspect-btn, data-preview, data-stats, charts,
 * preprocess-btn, preprocessing-output, add-family-features, create-model-btn,
 * model-summary, train-btn, training-status, threshold-slider, threshold-value,
 * confusion-matrix, performance-metrics, predict-btn, prediction-output, export-btn,
 * export-status.
 */

// -------------------------
// Global variables
// -------------------------
let trainData = null;
let testData = null;

let preprocessedTrainData = null;
let preprocessedTestData = null;

let model = null;
let trainingHistory = null;

let validationData = null;
let validationLabels = null;
let validationPredictions = null;

let testPredictions = null;

// For consistent normalization
let _normStats = {
  ageMedian: 0,
  fareMedian: 0,
  ageStd: 1,
  fareStd: 1,
  embarkedMode: 'S'
};

// For interpretability
let _featureNames = [];         // ordered feature names corresponding to input columns
let _gateLayerName = 'sigmoid_gate';

// -------------------------
// Schema configuration - change these for different datasets
// -------------------------
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Titanic categories (for one-hot)
const PCLASS_CATS = [1, 2, 3];
const SEX_CATS = ['male', 'female'];
const EMBARKED_CATS = ['C', 'Q', 'S'];

// -------------------------
// Small helpers (DOM safety)
// -------------------------
function $(id) {
  const el = document.getElementById(id);
  return el || null;
}
function setHTML(id, html) {
  const el = $(id);
  if (el) el.innerHTML = html;
}
function ensureEl(id, parentId, tag = 'div') {
  let el = $(id);
  if (el) return el;
  const parent = $(parentId);
  if (!parent) return null;
  el = document.createElement(tag);
  el.id = id;
  parent.appendChild(el);
  return el;
}

// -------------------------
// Robust CSV parser (handles quoted fields + commas)
// -------------------------
/**
 * parseCSV(csvText)
 * - Handles commas inside quotes.
 * - Handles escaped quotes "" inside quoted fields.
 * - Handles CRLF / LF.
 * - Returns array of objects.
 */
function parseCSV(csvText) {
  if (typeof csvText !== 'string') throw new Error('CSV text is not a string.');
  const rows = csvToRows(csvText);
  if (rows.length < 2) throw new Error('CSV has no data rows.');

  const headers = rows[0].map(h => (h ?? '').trim());
  const dataRows = rows.slice(1).filter(r => r.some(v => (v ?? '').trim() !== ''));

  return dataRows.map((row) => {
    const obj = {};
    for (let i = 0; i < headers.length; i++) {
      const key = headers[i];
      let raw = row[i] ?? '';
      raw = (typeof raw === 'string') ? raw.trim() : raw;

      // Normalize empties
      if (raw === '' || raw === 'NULL' || raw === 'null' || raw === undefined) {
        obj[key] = null;
        continue;
      }

      // Try numeric conversion safely
      // Only convert if it looks like a number (avoid converting "male"/"C"/etc.)
      const asNum = Number(raw);
      if (Number.isFinite(asNum) && String(raw).match(/^-?\d+(\.\d+)?$/)) {
        obj[key] = asNum;
      } else {
        obj[key] = raw;
      }
    }
    return obj;
  });
}

/**
 * csvToRows: returns array of rows, each row array of cell strings.
 * RFC4180-ish, enough for Kaggle Titanic.
 */
function csvToRows(text) {
  // Normalize newlines
  const s = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];

    if (inQuotes) {
      if (ch === '"') {
        // Escaped quote
        if (s[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
      continue;
    }

    // not in quotes
    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
    } else if (ch === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }

  // last cell
  row.push(cell);
  rows.push(row);

  // Remove trailing empty row if file ends with newline
  if (rows.length > 0 && rows[rows.length - 1].every(v => (v ?? '').trim() === '')) {
    rows.pop();
  }
  return rows;
}

// -------------------------
// Load data from uploaded CSV files
// -------------------------
async function loadData() {
  const trainFile = $('train-file')?.files?.[0];
  const testFile = $('test-file')?.files?.[0];

  if (!trainFile || !testFile) {
    alert('Please upload both training and test CSV files.');
    return;
  }

  setHTML('data-status', 'Loading data...');

  try {
    const trainText = await readFile(trainFile);
    trainData = parseCSV(trainText);

    const testText = await readFile(testFile);
    testData = parseCSV(testText);

    setHTML(
      'data-status',
      `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`
    );

    const inspectBtn = $('inspect-btn');
    if (inspectBtn) inspectBtn.disabled = false;

  } catch (error) {
    setHTML('data-status', `Error loading data: ${error.message}`);
    console.error(error);
  }
}

// Read file as text
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

// -------------------------
// Inspect the loaded data
// -------------------------
function inspectData() {
  if (!trainData || trainData.length === 0) {
    alert('Please load data first.');
    return;
  }

  // Preview
  const previewDiv = $('data-preview');
  if (previewDiv) {
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
  }

  // Stats
  const statsDiv = $('data-stats');
  if (statsDiv) {
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';

    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
      const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
      const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
      missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
  }

  // Visualizations
  createVisualizations();

  // Add code summary (LLM-style)
  renderCodeSummary();

  const preprocessBtn = $('preprocess-btn');
  if (preprocessBtn) preprocessBtn.disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
  const table = document.createElement('table');

  // Header
  const headerRow = document.createElement('tr');
  Object.keys(data[0]).forEach(key => {
    const th = document.createElement('th');
    th.textContent = key;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  // Rows
  data.forEach(row => {
    const tr = document.createElement('tr');
    Object.keys(data[0]).forEach(k => {
      const value = row[k];
      const td = document.createElement('td');
      td.textContent = value !== null && value !== undefined ? String(value) : 'NULL';
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
  const chartsDiv = $('charts');
  if (chartsDiv) chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

  // Survival by Sex
  const survivalBySex = {};
  trainData.forEach(row => {
    if (row.Sex && row.Survived !== undefined && row.Survived !== null) {
      if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { survived: 0, total: 0 };
      survivalBySex[row.Sex].total++;
      if (row.Survived === 1) survivalBySex[row.Sex].survived++;
    }
  });

  const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
    x: sex,
    y: (stats.survived / stats.total) * 100
  }));

  tfvis.render.barchart(
    { name: 'Survival Rate by Sex', tab: 'Charts' },
    sexData,
    { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
  );

  // Survival by Pclass
  const survivalByPclass = {};
  trainData.forEach(row => {
    if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined && row.Survived !== null) {
      if (!survivalByPclass[row.Pclass]) survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
      survivalByPclass[row.Pclass].total++;
      if (row.Survived === 1) survivalByPclass[row.Pclass].survived++;
    }
  });

  const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
    x: `Class ${pclass}`,
    y: (stats.survived / stats.total) * 100
  }));

  tfvis.render.barchart(
    { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
    pclassData,
    { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
  );

  if (chartsDiv) {
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
  }
}

// -------------------------
// Preprocess
// -------------------------
function preprocessData() {
  if (!trainData || !testData) {
    alert('Please load data first.');
    return;
  }

  const outputDiv = $('preprocessing-output');
  if (outputDiv) outputDiv.innerHTML = 'Preprocessing data...';

  try {
    // Compute imputation + normalization stats ONCE (training-only)
    const ages = trainData.map(r => r.Age).filter(v => v !== null && v !== undefined);
    const fares = trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined);
    const embarkedVals = trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined);

    _normStats.ageMedian = calculateMedian(ages);
    _normStats.fareMedian = calculateMedian(fares);
    _normStats.embarkedMode = calculateMode(embarkedVals) ?? 'S';

    _normStats.ageStd = calculateStdDev(ages) || 1;
    _normStats.fareStd = calculateStdDev(fares) || 1;

    // Determine feature names in the exact order of extractFeatures()
    buildFeatureNames();

    // Training arrays
    preprocessedTrainData = { features: [], labels: [] };
    trainData.forEach(row => {
      const x = extractFeatures(row);
      const y = row[TARGET_FEATURE];
      if (y === null || y === undefined) return; // skip bad rows
      preprocessedTrainData.features.push(x);
      preprocessedTrainData.labels.push(Number(y));
    });

    // Test arrays
    preprocessedTestData = { features: [], passengerIds: [] };
    testData.forEach(row => {
      const x = extractFeatures(row);
      preprocessedTestData.features.push(x);
      preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
    });

    // Convert to tensors
    preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
    preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

    if (outputDiv) {
      outputDiv.innerHTML = `
        <p>Preprocessing completed!</p>
        <p>Feature count: ${_featureNames.length}</p>
        <p>Training features shape: [${preprocessedTrainData.features.shape}]</p>
        <p>Training labels shape: [${preprocessedTrainData.labels.shape}]</p>
        <p>Test features shape: [${preprocessedTestData.features.length}, ${_featureNames.length}]</p>
      `;
    }

    const createBtn = $('create-model-btn');
    if (createBtn) createBtn.disabled = false;

  } catch (error) {
    if (outputDiv) outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
    console.error(error);
  }
}

function buildFeatureNames() {
  const familyEnabled = !!$('add-family-features')?.checked;

  const names = [];
  // Standardized numericals
  names.push('Age_z', 'Fare_z', 'SibSp', 'Parch');

  // One-hot categorical
  PCLASS_CATS.forEach(c => names.push(`Pclass_${c}`));
  SEX_CATS.forEach(c => names.push(`Sex_${c}`));
  EMBARKED_CATS.forEach(c => names.push(`Embarked_${c}`));

  if (familyEnabled) {
    names.push('FamilySize', 'IsAlone');
  }

  _featureNames = names;
}

// Extract features with imputation + normalization
function extractFeatures(row) {
  const age = (row.Age !== null && row.Age !== undefined) ? Number(row.Age) : _normStats.ageMedian;
  const fare = (row.Fare !== null && row.Fare !== undefined) ? Number(row.Fare) : _normStats.fareMedian;

  const embarked = (row.Embarked !== null && row.Embarked !== undefined && row.Embarked !== '') ? row.Embarked : _normStats.embarkedMode;

  const ageZ = (age - _normStats.ageMedian) / (_normStats.ageStd || 1);
  const fareZ = (fare - _normStats.fareMedian) / (_normStats.fareStd || 1);

  const sib = Number(row.SibSp ?? 0) || 0;
  const par = Number(row.Parch ?? 0) || 0;

  const pclassOH = oneHotEncode(Number(row.Pclass), PCLASS_CATS);
  const sexOH = oneHotEncode(row.Sex, SEX_CATS);
  const embOH = oneHotEncode(embarked, EMBARKED_CATS);

  let features = [ageZ, fareZ, sib, par].concat(pclassOH, sexOH, embOH);

  if ($('add-family-features')?.checked) {
    const familySize = sib + par + 1;
    const isAlone = familySize === 1 ? 1 : 0;
    features.push(familySize, isAlone);
  }

  return features;
}

// -------------------------
// Stats helpers
// -------------------------
function calculateMedian(values) {
  if (!values || values.length === 0) return 0;
  const arr = values.map(Number).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (arr.length === 0) return 0;
  const half = Math.floor(arr.length / 2);
  return (arr.length % 2 === 0) ? (arr[half - 1] + arr[half]) / 2 : arr[half];
}

function calculateMode(values) {
  if (!values || values.length === 0) return null;
  const freq = new Map();
  let bestVal = null;
  let bestCnt = -1;
  for (const v of values) {
    const key = String(v);
    const cnt = (freq.get(key) || 0) + 1;
    freq.set(key, cnt);
    if (cnt > bestCnt) {
      bestCnt = cnt;
      bestVal = v;
    }
  }
  return bestVal;
}

function calculateStdDev(values) {
  const arr = values.map(Number).filter(v => Number.isFinite(v));
  if (arr.length === 0) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const varr = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  return Math.sqrt(varr);
}

function oneHotEncode(value, categories) {
  const enc = new Array(categories.length).fill(0);
  const idx = categories.indexOf(value);
  if (idx !== -1) enc[idx] = 1;
  return enc;
}

// -------------------------
// Sigmoid gate layer (feature importance)
// -------------------------
/**
 * A learnable sigmoid gate:
 * y = x * sigmoid(w)
 * - w is a trainable vector (shape [D])
 * - sigmoid(w) in (0,1) can be read as "soft feature on/off"
 *
 * We expose the learned gates after training as feature importance.
 */
class SigmoidGateLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportsMasking = true;
  }

  build(inputShape) {
    // inputShape: [batch, D]
    const dim = inputShape[inputShape.length - 1];
    this.w = this.addWeight(
      'gate_logits',
      [dim],
      'float32',
      tf.initializers.zeros()
    );
    this.built = true;
  }

  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    return tf.tidy(() => {
      const gates = tf.sigmoid(this.w.read()); // [D]
      // Broadcast multiply: [B,D] * [D]
      return x.mul(gates);
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    const base = super.getConfig();
    return { ...base };
  }

  static get className() {
    return 'SigmoidGateLayer';
  }
}
tf.serialization.registerClass(SigmoidGateLayer);

// -------------------------
// Create the model
// -------------------------
function createModel() {
  if (!preprocessedTrainData) {
    alert('Please preprocess data first.');
    return;
  }

  const inputDim = preprocessedTrainData.features.shape[1];

  // Functional API to insert custom gate layer
  const input = tf.input({ shape: [inputDim] });

  // Gate layer for interpretability
  const gated = new SigmoidGateLayer({ name: _gateLayerName }).apply(input);

  // Shallow classifier
  const h = tf.layers.dense({ units: 16, activation: 'relu' }).apply(gated);
  const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(h);

  model = tf.model({ inputs: input, outputs: out });

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Display summary
  const summaryDiv = $('model-summary');
  if (summaryDiv) {
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    let html = `<p>Input dim: ${inputDim}</p>`;
    html += `<p>Feature count: ${_featureNames.length}</p>`;
    html += `<ul>`;
    model.layers.forEach((layer, i) => {
      html += `<li>Layer ${i + 1}: ${layer.name} (${layer.getClassName?.() || 'Layer'}) - Output: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    html += `</ul>`;
    html += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += html;

    // Show feature list (optional but helpful)
    summaryDiv.innerHTML += `<details><summary>Feature order (input columns)</summary><pre>${_featureNames.join('\n')}</pre></details>`;
  }

  const trainBtn = $('train-btn');
  if (trainBtn) trainBtn.disabled = false;
}

// -------------------------
// Train the model (fix: callbacks + validation predictions shape)
// -------------------------
async function trainModel() {
  if (!model || !preprocessedTrainData) {
    alert('Please create model first.');
    return;
  }

  const statusDiv = $('training-status');
  if (statusDiv) statusDiv.innerHTML = 'Training model...';

  try {
    // Stratified split (simple implementation)
    const X = preprocessedTrainData.features;
    const y = preprocessedTrainData.labels;

    const yArr = y.arraySync();
    const idx0 = [];
    const idx1 = [];
    for (let i = 0; i < yArr.length; i++) {
      (yArr[i] === 1 ? idx1 : idx0).push(i);
    }
    shuffleInPlace(idx0);
    shuffleInPlace(idx1);

    const trainFrac = 0.8;
    const n0Train = Math.floor(idx0.length * trainFrac);
    const n1Train = Math.floor(idx1.length * trainFrac);

    const trainIdx = idx0.slice(0, n0Train).concat(idx1.slice(0, n1Train));
    const valIdx = idx0.slice(n0Train).concat(idx1.slice(n1Train));

    shuffleInPlace(trainIdx);
    shuffleInPlace(valIdx);

    const trainFeatures = tf.gather(X, tf.tensor1d(trainIdx, 'int32'));
    const trainLabels = tf.gather(y, tf.tensor1d(trainIdx, 'int32'));

    const valFeatures = tf.gather(X, tf.tensor1d(valIdx, 'int32'));
    const valLabels = tf.gather(y, tf.tensor1d(valIdx, 'int32'));

    validationData = valFeatures;
    validationLabels = valLabels;

    // IMPORTANT FIX:
    // The previous code overwrote callbacks and used 'acc' keys.
    // In TFJS the metric name is typically 'acc' or 'accuracy' depending on version,
    // but fitCallbacks expects correct keys. We'll request 'loss' and 'acc' safely by aliasing.
    const visCallbacks = tfvis.show.fitCallbacks(
      { name: 'Training Performance', tab: 'Training' },
      ['loss', 'acc', 'val_loss', 'val_acc'],
      { callbacks: ['onEpochEnd'] }
    );

    // Early stopping (manual)
    let bestVal = Infinity;
    let wait = 0;
    const patience = 5;

    trainingHistory = await model.fit(trainFeatures, trainLabels, {
      epochs: 50,
      batchSize: 32,
      validationData: [valFeatures, valLabels],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          // Update tfjs-vis plots
          await visCallbacks.onEpochEnd(epoch, logs);

          const loss = logs.loss;
          // logs may include either acc or accuracy, depend on tfjs version
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const vLoss = logs.val_loss;
          const vAcc = logs.val_acc ?? logs.val_accuracy ?? 0;

          if (statusDiv) {
            statusDiv.innerHTML =
              `Epoch ${epoch + 1}/50 - loss: ${loss.toFixed(4)}, acc: ${acc.toFixed(4)}, val_loss: ${vLoss.toFixed(4)}, val_acc: ${vAcc.toFixed(4)}`;
          }

          // Early stopping
          if (vLoss < bestVal - 1e-6) {
            bestVal = vLoss;
            wait = 0;
          } else {
            wait += 1;
            if (wait >= patience) {
              model.stopTraining = true;
              if (statusDiv) statusDiv.innerHTML += `<p>Early stopping triggered (patience=${patience}).</p>`;
            }
          }
        }
      }
    });

    if (statusDiv) statusDiv.innerHTML += '<p>Training completed!</p>';

    // Validation predictions
    validationPredictions = model.predict(validationData); // [N,1]

    // Enable threshold slider + metrics
    const slider = $('threshold-slider');
    if (slider) {
      slider.disabled = false;
      slider.removeEventListener('input', updateMetrics); // avoid duplicates
      slider.addEventListener('input', updateMetrics);
    }

    // Enable predict
    const predBtn = $('predict-btn');
    if (predBtn) predBtn.disabled = false;

    // Show sigmoid gate importances
    await renderGateImportances();

    // Initial metrics
    await updateMetrics();

  } catch (error) {
    if (statusDiv) statusDiv.innerHTML = `Error during training: ${error.message}`;
    console.error(error);
  }
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// -------------------------
// Update metrics based on threshold (fix: flatten preds)
// -------------------------
async function updateMetrics() {
  if (!validationPredictions || !validationLabels) return;

  const slider = $('threshold-slider');
  const threshold = slider ? parseFloat(slider.value) : 0.5;
  const tv = $('threshold-value');
  if (tv) tv.textContent = threshold.toFixed(2);

  // FIX: predictions are [N,1] => flatten to [N]
  const pred2d = validationPredictions.arraySync(); // [[p],[p],...]
  const predVals = pred2d.map(v => Array.isArray(v) ? v[0] : v);

  const trueVals = validationLabels.arraySync(); // [0/1]

  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (let i = 0; i < predVals.length; i++) {
    const prediction = predVals[i] >= threshold ? 1 : 0;
    const actual = trueVals[i];

    if (prediction === 1 && actual === 1) tp++;
    else if (prediction === 0 && actual === 0) tn++;
    else if (prediction === 1 && actual === 0) fp++;
    else if (prediction === 0 && actual === 1) fn++;
  }

  // Ensure evaluation containers exist (in case index.html missing)
  ensureEl('confusion-matrix', 'performance-metrics', 'div');
  ensureEl('performance-metrics', 'performance-metrics', 'div');

  // Confusion matrix
  const cmDiv = $('confusion-matrix');
  if (cmDiv) {
    cmDiv.innerHTML = `
      <table>
        <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
        <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
        <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
      </table>
    `;
  }

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;
  const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;

  // Metrics table (this is the “evaluation table not show up” fix: always write HTML)
  const metricsDiv = $('performance-metrics');
  if (metricsDiv) {
    metricsDiv.innerHTML = `
      <h4>Performance Metrics</h4>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Accuracy</td><td>${(accuracy * 100).toFixed(2)}%</td></tr>
        <tr><td>Precision</td><td>${precision.toFixed(4)}</td></tr>
        <tr><td>Recall</td><td>${recall.toFixed(4)}</td></tr>
        <tr><td>F1</td><td>${f1.toFixed(4)}</td></tr>
      </table>
    `;
  }

  await plotROC(trueVals, predVals);
}

// Plot ROC curve + AUC
async function plotROC(trueLabels, predictions) {
  const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
  const rocData = [];

  thresholds.forEach(threshold => {
    let tp = 0, fn = 0, fp = 0, tn = 0;

    for (let i = 0; i < predictions.length; i++) {
      const pred = predictions[i] >= threshold ? 1 : 0;
      const actual = trueLabels[i];

      if (actual === 1) {
        if (pred === 1) tp++;
        else fn++;
      } else {
        if (pred === 1) fp++;
        else tn++;
      }
    }

    const tpr = tp / (tp + fn) || 0;
    const fpr = fp / (fp + tn) || 0;
    rocData.push({ fpr, tpr });
  });

  // Sort by FPR to compute AUC properly
  rocData.sort((a, b) => a.fpr - b.fpr);

  let auc = 0;
  for (let i = 1; i < rocData.length; i++) {
    const dx = rocData[i].fpr - rocData[i - 1].fpr;
    const avgY = (rocData[i].tpr + rocData[i - 1].tpr) / 2;
    auc += dx * avgY;
  }
  auc = Math.max(0, Math.min(1, auc));

  // Render ROC in visor (tab Evaluation)
  tfvis.render.linechart(
    { name: 'ROC Curve', tab: 'Evaluation' },
    { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
    {
      xLabel: 'False Positive Rate',
      yLabel: 'True Positive Rate',
      width: 420,
      height: 420
    }
  );

  // Append AUC to metrics table
  const metricsDiv = $('performance-metrics');
  if (metricsDiv) {
    metricsDiv.innerHTML += `<p><b>AUC:</b> ${auc.toFixed(4)}</p>`;
  }
}

// -------------------------
// Predict on test data (fix: probability extraction + threshold uses slider)
// -------------------------
async function predict() {
  if (!model || !preprocessedTestData) {
    alert('Please train model first.');
    return;
  }

  const outputDiv = $('prediction-output');
  if (outputDiv) outputDiv.innerHTML = 'Making predictions...';

  try {
    const testFeatures = tf.tensor2d(preprocessedTestData.features);
    testPredictions = model.predict(testFeatures); // [N,1]

    const pred2d = testPredictions.arraySync();
    const probs = pred2d.map(v => Array.isArray(v) ? v[0] : v);

    const threshold = parseFloat($('threshold-slider')?.value ?? '0.5');

    const results = preprocessedTestData.passengerIds.map((id, i) => ({
      PassengerId: id,
      Survived: probs[i] >= threshold ? 1 : 0,
      Probability: probs[i]
    }));

    if (outputDiv) {
      outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
      outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
      outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;
    }

    const exportBtn = $('export-btn');
    if (exportBtn) exportBtn.disabled = false;

  } catch (error) {
    if (outputDiv) outputDiv.innerHTML = `Error during prediction: ${error.message}`;
    console.error(error);
  }
}

// Create prediction table
function createPredictionTable(data) {
  const table = document.createElement('table');

  const headerRow = document.createElement('tr');
  ['PassengerId', 'Survived', 'Probability'].forEach(header => {
    const th = document.createElement('th');
    th.textContent = header;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  data.forEach(row => {
    const tr = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(key => {
      const td = document.createElement('td');
      td.textContent = key === 'Probability' ? Number(row[key]).toFixed(4) : row[key];
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  return table;
}

// -------------------------
// Export results (fix: probabilities extraction + threshold uses slider)
// -------------------------
async function exportResults() {
  if (!testPredictions || !preprocessedTestData) {
    alert('Please make predictions first.');
    return;
  }

  const statusDiv = $('export-status');
  if (statusDiv) statusDiv.innerHTML = 'Exporting results...';

  try {
    const pred2d = testPredictions.arraySync();
    const probs = pred2d.map(v => Array.isArray(v) ? v[0] : v);

    const threshold = parseFloat($('threshold-slider')?.value ?? '0.5');

    let submissionCSV = 'PassengerId,Survived\n';
    preprocessedTestData.passengerIds.forEach((id, i) => {
      submissionCSV += `${id},${probs[i] >= threshold ? 1 : 0}\n`;
    });

    let probabilitiesCSV = 'PassengerId,Probability\n';
    preprocessedTestData.passengerIds.forEach((id, i) => {
      probabilitiesCSV += `${id},${probs[i].toFixed(6)}\n`;
    });

    const submissionLink = document.createElement('a');
    submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
    submissionLink.download = 'submission.csv';

    const probabilitiesLink = document.createElement('a');
    probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
    probabilitiesLink.download = 'probabilities.csv';

    submissionLink.click();
    probabilitiesLink.click();

    await model.save('downloads://titanic-tfjs-model');

    if (statusDiv) {
      statusDiv.innerHTML = `
        <p>Export completed!</p>
        <p>Downloaded: submission.csv (Kaggle submission format)</p>
        <p>Downloaded: probabilities.csv (Prediction probabilities)</p>
        <p>Model saved to browser downloads</p>
      `;
    }
  } catch (error) {
    if (statusDiv) statusDiv.innerHTML = `Error during export: ${error.message}`;
    console.error(error);
  }
}

// -------------------------
// Homework item #3: LLM-style code summary (in-app)
// -------------------------
function renderCodeSummary() {
  // Put summary under data-stats (non-invasive)
  const statsDiv = $('data-stats');
  if (!statsDiv) return;

  const summary = `
    <details style="margin-top:10px;">
      <summary><b>LLM-style summary: what this app.js does</b></summary>
      <ol>
        <li><b>Load CSVs</b>: reads train.csv and test.csv from file inputs, parses with a robust CSV parser that respects quotes/commas.</li>
        <li><b>Inspect</b>: shows a preview table, dataset shape, missingness per column, and tfjs-vis bar charts for survival rates.</li>
        <li><b>Preprocess</b>: imputes Age/Fare/Embarked from training stats, standardizes Age/Fare, one-hot encodes Pclass/Sex/Embarked, optionally adds FamilySize and IsAlone, then builds tensors.</li>
        <li><b>Model</b>: uses a shallow neural net. Before the hidden layer it applies a <b>sigmoid gate</b> (learnable per-feature multipliers) for interpretability.</li>
        <li><b>Train</b>: stratified 80/20 split, tfjs-vis live plots, manual early stopping, then predicts on validation set.</li>
        <li><b>Evaluate</b>: threshold slider updates confusion matrix + precision/recall/F1/accuracy, plots ROC curve and computes AUC.</li>
        <li><b>Predict & Export</b>: predicts on test.csv, creates submission.csv + probabilities.csv, downloads them, saves model.</li>
      </ol>
    </details>
  `;
  statsDiv.innerHTML += summary;
}

// -------------------------
// Homework item #4: show sigmoid gate importances
// -------------------------
async function renderGateImportances() {
  if (!model) return;

  const gateLayer = model.getLayer(_gateLayerName);
  if (!gateLayer) return;

  // Read trainable logits -> sigmoid -> importance in (0,1)
  const w = gateLayer.getWeights?.()[0]; // gate_logits
  if (!w) return;

  const gates = tf.sigmoid(w).arraySync(); // [D]

  // Pair with feature names
  const pairs = gates.map((g, i) => ({
    feature: _featureNames[i] ?? `f_${i}`,
    gate: g
  }));

  pairs.sort((a, b) => b.gate - a.gate);

  // Render into model-summary
  const summaryDiv = $('model-summary');
  if (!summaryDiv) return;

  const topN = Math.min(20, pairs.length);
  let html = `<h4>Sigmoid Gate Feature Importance (top ${topN})</h4>`;
  html += `<p>Gate values are in (0,1). Higher means the model keeps more of that feature signal.</p>`;
  html += `<table><tr><th>Feature</th><th>Gate</th></tr>`;
  for (let i = 0; i < topN; i++) {
    html += `<tr><td>${pairs[i].feature}</td><td>${pairs[i].gate.toFixed(4)}</td></tr>`;
  }
  html += `</table>`;

  summaryDiv.innerHTML += html;

  // Also show a bar chart in visor
  tfvis.render.barchart(
    { name: 'Sigmoid Gate Importances', tab: 'Interpretability' },
    pairs.slice(0, topN).map(p => ({ x: p.feature, y: p.gate })),
    { xLabel: 'Feature', yLabel: 'Gate (0..1)' }
  );
}

// -------------------------
// Expose functions to window for onclick handlers
// -------------------------
window.loadData = loadData;
window.inspectData = inspectData;
window.preprocessData = preprocessData;
window.createModel = createModel;
window.trainModel = trainModel;
window.predict = predict;
window.exportResults = exportResults;
