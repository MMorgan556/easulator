// Calculator core: input state + a small precedence-aware evaluator (no eval()).
(function (root) {
  'use strict';

  const OPS = {
    '+': { prec: 1, fn: (a, b) => a + b },
    '-': { prec: 1, fn: (a, b) => a - b },
    '×': { prec: 2, fn: (a, b) => a * b },
    '÷': {
      prec: 2,
      fn: (a, b) => {
        if (b === 0) throw new Error('Cannot divide by zero');
        return a / b;
      },
    },
  };

  // Evaluate an array of alternating numbers and operator symbols.
  function evaluate(tokens) {
    const values = [];
    const ops = [];
    const apply = () => {
      const b = values.pop();
      const a = values.pop();
      values.push(OPS[ops.pop()].fn(a, b));
    };
    for (const t of tokens) {
      if (typeof t === 'number') {
        values.push(t);
      } else {
        while (ops.length && OPS[ops[ops.length - 1]].prec >= OPS[t].prec) apply();
        ops.push(t);
      }
    }
    while (ops.length) apply();
    return values[0];
  }

  // Trim floating-point noise (0.1 + 0.2 -> 0.3) and keep output readable.
  function format(n) {
    if (!Number.isFinite(n)) throw new Error('Result is too large');
    const rounded = parseFloat(n.toPrecision(12));
    const abs = Math.abs(rounded);
    if (abs !== 0 && (abs >= 1e12 || abs < 1e-9)) return rounded.toExponential(6).replace(/\.?0+e/, 'e');
    return String(rounded);
  }

  class Calculator {
    constructor() {
      this.clear();
    }

    clear() {
      this.tokens = [];
      this.current = '0';
      this.justEvaluated = false;
      this.error = null;
    }

    // Start a fresh number: after an operator, a result, or an error.
    _beginEntry() {
      if (this.error || this.justEvaluated) this.clear();
      if (this.current === null) this.current = '0';
    }

    inputDigit(d) {
      this._beginEntry();
      if (this.current.replace(/[-.]/g, '').length >= 15) return;
      this.current = this.current === '0' ? d : this.current === '-0' ? '-' + d : this.current + d;
    }

    inputDecimal() {
      this._beginEntry();
      if (!this.current.includes('.')) this.current += '.';
    }

    inputOperator(op) {
      if (!OPS[op]) return;
      if (this.error) this.clear();
      this.justEvaluated = false;
      // Pressing an operator right after another one replaces it.
      if (this.current === null) {
        this.tokens[this.tokens.length - 1] = op;
        return;
      }
      this.tokens.push(parseFloat(this.current), op);
      this.current = null;
    }

    toggleSign() {
      if (this.error) return;
      this.justEvaluated = false;
      if (this.current === null) this.current = '0';
      this.current = this.current.startsWith('-') ? this.current.slice(1) : '-' + this.current;
    }

    percent() {
      if (this.error || this.current === null) return;
      this.current = format(parseFloat(this.current) / 100);
    }

    backspace() {
      if (this.error || this.justEvaluated) return this.clear();
      if (this.current === null) {
        // Remove the trailing operator and resume editing the previous number.
        this.tokens.pop();
        this.current = String(this.tokens.pop());
        return;
      }
      const next = this.current.slice(0, -1);
      this.current = next === '' || next === '-' ? '0' : next;
    }

    equals() {
      if (this.error) return;
      const tokens = this.tokens.slice();
      if (this.current === null) tokens.pop(); // ignore a dangling operator
      else tokens.push(parseFloat(this.current));
      if (tokens.length === 0) return;
      try {
        this.current = format(evaluate(tokens));
        this.tokens = [];
        this.justEvaluated = true;
      } catch (e) {
        this.tokens = [];
        this.current = null;
        this.error = e.message;
      }
    }

    // Text for the small line above the main display.
    get expression() {
      const parts = this.tokens.map((t) => (typeof t === 'number' ? format(t) : t));
      return parts.join(' ');
    }

    get display() {
      if (this.error) return this.error;
      return this.current === null ? format(this.tokens[this.tokens.length - 2]) : this.current;
    }
  }

  const api = { Calculator, evaluate, format };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CalculatorCore = api;
})(this);
