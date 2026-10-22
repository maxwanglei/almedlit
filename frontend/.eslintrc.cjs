module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
  },
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: "latest",
    sourceType: "module",
    project: "./tsconfig.json",
    tsconfigRootDir: __dirname,
  },
  plugins: ["@typescript-eslint", "react-hooks"],
  extends: ["eslint:recommended", "plugin:@typescript-eslint/recommended"],
  ignorePatterns: ["dist", "node_modules"],
  overrides: [{
    files: ["e2e/**/*.ts", "playwright.config.ts"],
    env: { node: true },
    parserOptions: { project: "./tsconfig.e2e.json" },
  }],
  rules: {
    "@typescript-eslint/consistent-type-imports": "error",
    "react-hooks/exhaustive-deps": "warn",
    "react-hooks/rules-of-hooks": "error",
  },
};
