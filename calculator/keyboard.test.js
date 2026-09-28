// Keyboard shortcut tests: runs the real app.js against the buttons in index.html
// using a tiny fake DOM, so no browser or npm packages are needed.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const dir = __dirname;
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');

function makeElement(attrs = {}) {
  const classes = new Set((attrs.class || '').split(/\s+/).filter(Boolean));
  return {
    dataset: {},
    textContent: '',
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
      contains: (c) => classes.has(c),
    },
    closest(sel) {
      return sel === 'button' && this.isButton ? this : null;
    },
  };
}

// Build a page from index.html's buttons, load app.js, and return a `type` helper.
function openApp() {
  const listeners = { keys: [], document: [] };
  const buttons = [...html.matchAll(/<button([^>]*)>/g)].map(([, attrs]) => {
    const el = makeElement({ class: (attrs.match(/class="([^"]*)"/) || [])[1] });
    el.isButton = true;
    for (const [, name, value] of attrs.matchAll(/data-(\w+)="([^"]*)"/g)) el.dataset[name] = value;
    el.click = () => listeners.keys.forEach((fn) => fn({ target: el }));
    return el;
  });
  const byId = { expression: makeElement(), current: makeElement() };
  const keys = { addEventListener: (type, fn) => type === 'click' && listeners.keys.push(fn) };

  const document = {
    getElementById: (id) => byId[id],
    querySelector(sel) {
      if (sel === '.keys') return keys;
      const m = sel.match(/^\[data-(\w+)="(.*)"\]$/);
      return m ? buttons.find((b) => b.dataset[m[1]] === m[2]) || null : null;
    },
    querySelectorAll(sel) {
      const m = sel.match(/^\[data-(\w+)\]$/);
      return m ? buttons.filter((b) => m[1] in b.dataset) : [];
    },
    addEventListener: (type, fn) => type === 'keydown' && listeners.document.push(fn),
  };

  const context = vm.createContext({ document, setTimeout: () => {} });
  for (const file of ['calculator.js', 'app.js']) {
    vm.runInContext(fs.readFileSync(path.join(dir, file), 'utf8'), context, { filename: file });
  }

  const display = () => byId.current.textContent;
  // Press each key in turn, like a user typing. Returns the main display text.
  function type(...keyNames) {
    for (const key of keyNames) {
      const event = { key, ctrlKey: false, metaKey: false, altKey: false, defaultPrevented: false };
      event.preventDefault = () => (event.defaultPrevented = true);
      listeners.document.forEach((fn) => fn(event));
    }
    return display();
  }
  // Split a string like "12+3" into single-character keys.
  const typeChars = (s) => type(...s);

  return { type, typeChars, display, expression: () => byId.expression.textContent, buttons, listeners };
}

test('digit keys enter numbers', () => {
  const app = openApp();
  assert.equal(app.display(), '0');
  assert.equal(app.typeChars('1234567890'), '1234567890');
});

test('+ - * / keys do arithmetic', () => {
  assert.equal(openApp().type('7', '+', '5', 'Enter'), '12');
  assert.equal(openApp().type('7', '-', '9', 'Enter'), '-2');
  assert.equal(openApp().type('6', '*', '7', 'Enter'), '42');
  assert.equal(openApp().type('9', '/', '4', 'Enter'), '2.25');
  assert.equal(openApp().type('6', 'x', '7', '='), '42');
});

test('operator keys show the pending expression', () => {
  const app = openApp();
  app.type('1', '2', '*');
  assert.equal(app.expression(), '12 ×');
  const times = app.buttons.find((b) => b.dataset.op === '×');
  assert.ok(times.classList.contains('active'));
});

test('Enter and = both evaluate', () => {
  assert.equal(openApp().typeChars('2+3*4='), '14');
  assert.equal(openApp().type('2', '+', '3', '*', '4', 'Enter'), '14');
});

test('Backspace deletes the last digit', () => {
  const app = openApp();
  assert.equal(app.type('1', '2', '3', 'Backspace'), '12');
  assert.equal(app.type('Backspace', 'Backspace'), '0');
});

test('Escape and Delete clear everything', () => {
  const app = openApp();
  app.typeChars('12+34');
  assert.equal(app.type('Escape'), '0');
  assert.equal(app.expression(), '');
  app.typeChars('56*');
  assert.equal(app.type('Delete'), '0');
  assert.equal(app.typeChars('2='), '2');
});

test('decimal and percent keys', () => {
  assert.equal(openApp().typeChars('0.1+0.2='), '0.3');
  assert.equal(openApp().typeChars('1,5*2='), '3');
  assert.equal(openApp().typeChars('50%'), '0.5');
});

test('dividing by zero from the keyboard shows an error, and Escape recovers', () => {
  const app = openApp();
  assert.equal(app.typeChars('8/0='), 'Cannot divide by zero');
  assert.equal(app.type('Escape'), '0');
  assert.equal(app.typeChars('8/2='), '4');
});

test('handled keys prevent the browser default; others are ignored', () => {
  const app = openApp();
  const send = (key, mods = {}) => {
    const event = { key, ctrlKey: false, metaKey: false, altKey: false, ...mods, defaultPrevented: false };
    event.preventDefault = () => (event.defaultPrevented = true);
    app.listeners.document.forEach((fn) => fn(event));
    return event.defaultPrevented;
  };
  assert.equal(send('5'), true);
  assert.equal(send('Enter'), true);
  assert.equal(send('a'), false);
  assert.equal(send('Tab'), false);
  // Browser shortcuts like Cmd+R or Ctrl+5 pass through untouched.
  assert.equal(send('r', { metaKey: true }), false);
  assert.equal(send('5', { ctrlKey: true }), false);
  assert.equal(app.display(), '5');
});
