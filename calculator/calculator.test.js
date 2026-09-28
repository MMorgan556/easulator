const test = require('node:test');
const assert = require('node:assert/strict');
const { Calculator, evaluate, format } = require('./calculator.js');

function press(keys) {
  const c = new Calculator();
  for (const k of keys.split(' ')) {
    if (/^\d+$/.test(k)) [...k].forEach((d) => c.inputDigit(d));
    else if (k === '.') c.inputDecimal();
    else if (k === '=') c.equals();
    else if (k === '±') c.toggleSign();
    else if (k === '%') c.percent();
    else if (k === '⌫') c.backspace();
    else c.inputOperator(k);
  }
  return c;
}

test('respects operator precedence', () => {
  assert.equal(evaluate([2, '+', 3, '×', 4]), 14);
  assert.equal(evaluate([10, '-', 4, '-', 3]), 3);
  assert.equal(evaluate([100, '÷', 10, '÷', 2]), 5);
});

test('basic key sequences', () => {
  assert.equal(press('2 + 3 × 4 =').display, '14');
  assert.equal(press('7 ÷ 2 =').display, '3.5');
  assert.equal(press('0 . 1 + 0 . 2 =').display, '0.3');
});

test('replaces a repeated operator and ignores a dangling one', () => {
  assert.equal(press('5 + × 3 =').display, '15');
  assert.equal(press('5 + =').display, '5');
});

test('sign, percent and backspace', () => {
  assert.equal(press('5 ± × 3 =').display, '-15');
  assert.equal(press('50 %').display, '0.5');
  assert.equal(press('123 ⌫').display, '12');
  assert.equal(press('12 + ⌫ 3 =').display, '123');
});

test('division by zero shows an error, then recovers', () => {
  const c = press('5 ÷ 0 =');
  assert.equal(c.display, 'Cannot divide by zero');
  c.inputDigit('4');
  assert.equal(c.display, '4');
});

test('typing after a result starts a new number; operator continues', () => {
  assert.equal(press('2 + 2 = 9').display, '9');
  assert.equal(press('2 + 2 = × 3 =').display, '12');
});

test('formats large and tiny numbers', () => {
  assert.equal(format(1e15), '1e+15');
  assert.equal(format(123456.789), '123456.789');
});
