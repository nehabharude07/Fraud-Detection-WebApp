const form = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');

const resultEmpty = document.getElementById('result-empty');
const resultContent = document.getElementById('result-content');
const resultError = document.getElementById('result-error');
const errorText = document.getElementById('error-text');

const gaugeFill = document.getElementById('gauge-fill');
const gaugeValue = document.getElementById('gauge-value');
const verdictBadge = document.getElementById('verdict-badge');
const verdictText = document.getElementById('verdict-text');
const breakdownList = document.getElementById('breakdown-list');

const GAUGE_CIRCUMFERENCE = 251.2; // half-circle path length approximation used in CSS

const RISK_COPY = {
  low: 'Balances and transfer pattern line up with routine activity. No action needed.',
  medium: 'Some signals look unusual. Worth a second look before approving.',
  high: 'Strong resemblance to confirmed fraud patterns. Hold this transaction for review.',
  critical: 'Very high confidence of fraud — matches classic account-draining behavior.',
};

const RISK_LABEL = {
  low: 'Low risk',
  medium: 'Medium risk',
  high: 'High risk',
  critical: 'Critical risk',
};

function fmtMoney(n) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(n);
}

function setGauge(probability, risk) {
  const offset = GAUGE_CIRCUMFERENCE * (1 - probability);
  gaugeFill.style.strokeDashoffset = offset;
  const colors = { low: '#3ddc97', medium: '#ffb454', high: '#ff6b5e', critical: '#ff6b5e' };
  gaugeFill.style.stroke = colors[risk] || '#3ddc97';
  gaugeValue.textContent = `${(probability * 100).toFixed(1)}%`;
}

function renderBreakdown(values) {
  breakdownList.innerHTML = '';
  const rows = [
    ['Type', values.type.replace('_', ' ')],
    ['Amount', fmtMoney(values.amount)],
    ['Sender balance change', `${fmtMoney(values.oldbalanceOrg)} → ${fmtMoney(values.newbalanceOrig)}`],
    ['Recipient balance change', `${fmtMoney(values.oldbalanceDest)} → ${fmtMoney(values.newbalanceDest)}`],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value;
    breakdownList.append(dt, dd);
  }
}

async function scoreTransaction(payload) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Something went wrong scoring this transaction.');
  return data;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;

  const values = {
    type: form.type.value,
    amount: form.amount.value,
    oldbalanceOrg: form.oldbalanceOrg.value,
    newbalanceOrig: form.newbalanceOrig.value,
    oldbalanceDest: form.oldbalanceDest.value,
    newbalanceDest: form.newbalanceDest.value,
    step: form.step.value || 1,
  };

  resultError.classList.add('hidden');

  try {
    const data = await scoreTransaction(values);
    const risk = data.riskLevel;

    resultEmpty.classList.add('hidden');
    resultContent.classList.remove('hidden');

    setGauge(data.fraudProbability, risk);
    verdictBadge.textContent = RISK_LABEL[risk] || risk;
    verdictBadge.dataset.risk = risk;
    verdictText.textContent = RISK_COPY[risk] || '';
    renderBreakdown({ ...values, amount: Number(values.amount) });

    window.lastPredictionContext = {
      ...values,
      amount: Number(values.amount),
      fraudProbability: data.fraudProbability,
      riskLevel: risk,
    };
  } catch (err) {
    resultEmpty.classList.add('hidden');
    resultContent.classList.add('hidden');
    resultError.classList.remove('hidden');
    errorText.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
  }
});

const SAMPLES = {
  legit: {
    type: 'PAYMENT',
    amount: 4823.50,
    oldbalanceOrg: 18250.00,
    newbalanceOrig: 13426.50,
    oldbalanceDest: 0,
    newbalanceDest: 0,
    step: 12,
  },
  fraud: {
    type: 'TRANSFER',
    amount: 182450.00,
    oldbalanceOrg: 182450.00,
    newbalanceOrig: 0,
    oldbalanceDest: 0,
    newbalanceDest: 0,
    step: 9,
  },
};

document.querySelectorAll('.quickfill__btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const sample = SAMPLES[btn.dataset.sample];
    for (const [key, value] of Object.entries(sample)) {
      if (form[key]) form[key].value = value;
    }
  });
});

// ---------------------------------------------------------------------------
// Receipt photo: upload, camera capture, preview, optional AI extraction
// ---------------------------------------------------------------------------

const receiptUpload = document.getElementById('receipt-upload');
const receiptCamera = document.getElementById('receipt-camera');
const receiptImg = document.getElementById('receipt-img');
const receiptPlaceholder = document.getElementById('receipt-placeholder');
const receiptRemove = document.getElementById('receipt-remove');
const receiptExtractBtn = document.getElementById('receipt-extract');
const receiptNote = document.getElementById('receipt-note');

let receiptDataUrl = null;

function handleReceiptFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    receiptDataUrl = reader.result;
    receiptImg.src = receiptDataUrl;
    receiptImg.classList.remove('hidden');
    receiptPlaceholder.classList.add('hidden');
    receiptRemove.classList.remove('hidden');
    receiptExtractBtn.disabled = !window.LEDGER_AI_ENABLED;
    receiptNote.textContent = window.LEDGER_AI_ENABLED
      ? ''
      : 'Photo attached. Turn on AI features to auto-extract the amount.';
    receiptNote.classList.remove('is-error');
  };
  reader.readAsDataURL(file);
}

receiptUpload.addEventListener('change', (e) => handleReceiptFile(e.target.files[0]));
receiptCamera.addEventListener('change', (e) => handleReceiptFile(e.target.files[0]));

receiptRemove.addEventListener('click', () => {
  receiptDataUrl = null;
  receiptImg.src = '';
  receiptImg.classList.add('hidden');
  receiptPlaceholder.classList.remove('hidden');
  receiptRemove.classList.add('hidden');
  receiptExtractBtn.disabled = true;
  receiptNote.textContent = '';
  receiptUpload.value = '';
  receiptCamera.value = '';
});

receiptExtractBtn.addEventListener('click', async () => {
  if (!receiptDataUrl) return;
  receiptExtractBtn.disabled = true;
  receiptNote.textContent = 'Reading the receipt…';
  receiptNote.classList.remove('is-error');

  const [header, base64] = receiptDataUrl.split(',');
  const mediaType = header.match(/data:(.*);base64/)?.[1] || 'image/jpeg';

  try {
    const res = await fetch('/api/analyze-receipt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64, mediaType }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not read the receipt.');

    if (data.amount) form.amount.value = data.amount;
    if (data.suggestedType && form.type.querySelector(`option[value="${data.suggestedType}"]`)) {
      form.type.value = data.suggestedType;
    }

    const parts = [];
    if (data.merchant) parts.push(`Merchant: ${data.merchant}`);
    if (data.date) parts.push(`Date: ${data.date}`);
    if (data.notes) parts.push(data.notes);
    receiptNote.textContent = parts.length
      ? parts.join(' · ')
      : 'Read the receipt, but couldn\u2019t find a clear amount — fill it in manually.';
  } catch (err) {
    receiptNote.textContent = err.message;
    receiptNote.classList.add('is-error');
  } finally {
    receiptExtractBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Chat assistant
// ---------------------------------------------------------------------------

const chatToggle = document.getElementById('chat-toggle');
const chatPanel = document.getElementById('chat-panel');
const chatClose = document.getElementById('chat-close');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');

let chatHistory = [];
let lastPredictionContext = null;

function addChatMessage(text, role) {
  const div = document.createElement('div');
  div.className = `chat-msg chat-msg--${role}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatToggle.addEventListener('click', () => {
  chatPanel.classList.toggle('hidden');
  if (!chatPanel.classList.contains('hidden')) chatInput.focus();
});
chatClose.addEventListener('click', () => chatPanel.classList.add('hidden'));

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addChatMessage(message, 'user');
  chatInput.value = '';

  if (!window.LEDGER_AI_ENABLED) {
    addChatMessage(
      'Chat isn\u2019t turned on yet — set an ANTHROPIC_API_KEY environment variable and restart the app to enable it.',
      'error'
    );
    return;
  }

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        context: lastPredictionContext,
        history: chatHistory,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong.');

    addChatMessage(data.reply, 'bot');
    chatHistory.push({ role: 'user', content: message });
    chatHistory.push({ role: 'assistant', content: data.reply });
    if (chatHistory.length > 12) chatHistory = chatHistory.slice(-12);
  } catch (err) {
    addChatMessage(err.message, 'error');
  }
});