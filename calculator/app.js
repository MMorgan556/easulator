(function () {
  'use strict';

  const calc = new CalculatorCore.Calculator();
  const expressionEl = document.getElementById('expression');
  const currentEl = document.getElementById('current');
  const opButtons = document.querySelectorAll('[data-op]');
  const clearButton = document.querySelector('[data-action="clear"]');

  function render() {
    const text = calc.display;
    currentEl.textContent = text;
    currentEl.classList.toggle('error', Boolean(calc.error));
    currentEl.classList.toggle('small', !calc.error && text.length > 9);
    expressionEl.textContent = calc.expression;

    // Highlight the pending operator until the next number is entered.
    const pending = calc.current === null ? calc.tokens[calc.tokens.length - 1] : null;
    opButtons.forEach((b) => b.classList.toggle('active', b.dataset.op === pending));
    clearButton.textContent = calc.current !== '0' || calc.tokens.length ? 'C' : 'AC';
  }

  const actions = {
    clear: () => calc.clear(),
    sign: () => calc.toggleSign(),
    percent: () => calc.percent(),
    backspace: () => calc.backspace(),
    decimal: () => calc.inputDecimal(),
    equals: () => calc.equals(),
  };

  document.querySelector('.keys').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (btn.dataset.digit) calc.inputDigit(btn.dataset.digit);
    else if (btn.dataset.op) calc.inputOperator(btn.dataset.op);
    else actions[btn.dataset.action]();
    render();
  });

  const keyMap = {
    '+': '[data-op="+"]',
    '-': '[data-op="-"]',
    '*': '[data-op="×"]',
    x: '[data-op="×"]',
    '/': '[data-op="÷"]',
    '.': '[data-action="decimal"]',
    ',': '[data-action="decimal"]',
    '%': '[data-action="percent"]',
    Enter: '[data-action="equals"]',
    '=': '[data-action="equals"]',
    Backspace: '[data-action="backspace"]',
    Escape: '[data-action="clear"]',
    Delete: '[data-action="clear"]',
  };

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const selector = /^[0-9]$/.test(e.key) ? `[data-digit="${e.key}"]` : keyMap[e.key];
    if (!selector) return;
    e.preventDefault();
    const btn = document.querySelector(selector);
    btn.click();
    btn.classList.add('pressed');
    setTimeout(() => btn.classList.remove('pressed'), 100);
  });

  render();
})();
