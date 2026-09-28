# Calculator

A small calculator that runs in the browser. It has no dependencies and needs no build step.

Open `index.html` in a browser to use it.

- Handles order of operations: `2 + 3 × 4 = 14`
- Has ±, %, backspace and clear (AC/C) keys
- Rounds away floating-point noise: `0.1 + 0.2 = 0.3`
- Shows an error when you divide by zero
- Works with the keyboard: digits, `+ - * /`, `.`, `%`, `Enter`/`=`, `Backspace`, and `Esc` to clear
- Follows the system's light or dark mode

The logic is in `calculator.js` and does not use `eval`. `app.js` connects it to the page.

Run the tests with Node 18 or later. `calculator.test.js` covers the math and `keyboard.test.js` runs `app.js` against the page's buttons to check the keyboard shortcuts:

```sh
node --test calculator/*.test.js
```
