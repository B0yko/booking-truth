import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["node_modules/**", ".venv/**", "results/**", "runs/**"] },
  js.configs.recommended,
  {
    files: ["widget/widget.js"],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "script",
      globals: { ...globals.browser, module: "readonly" },
    },
    rules: {
      "no-var": "error",
      "prefer-const": "error",
      eqeqeq: ["error", "always", { null: "ignore" }],
      "no-implicit-globals": "error",
      "no-restricted-syntax": [
        "error",
        { selector: "MemberExpression[property.name='innerHTML']", message: "Build DOM nodes; never parse HTML." },
        { selector: "MemberExpression[property.name='outerHTML']", message: "Build DOM nodes; never parse HTML." },
        { selector: "CallExpression[callee.property.name='insertAdjacentHTML']", message: "Build DOM nodes." },
        { selector: "CallExpression[callee.name='eval']", message: "No eval." },
      ],
    },
  },
  {
    files: ["tests/widget/**/*.mjs", "eslint.config.mjs"],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      globals: { ...globals.node },
    },
    rules: { "prefer-const": "error", "no-var": "error" },
  },
];
