import { test as base, expect, type BrowserContext, type Response, type TestInfo } from "@playwright/test";

interface ApiResponseLedger {
  unexpected: string[];
  listeners: Map<BrowserContext, (response: Response) => void>;
}
const ledgers = new WeakMap<TestInfo, ApiResponseLedger>();

export function assertNoUnexpectedApiResponses(requests: readonly string[]): void {
  expect(requests, "Unexpected API responses reached a shared fixture").toEqual([]);
}

export function registerMockContext(context: BrowserContext): void {
  const ledger = ledgers.get(base.info());
  if (!ledger || ledger.listeners.has(context)) return;
  const record = (response: Response): void => {
    const url = new URL(response.url());
    if (url.pathname.startsWith("/api/") && [403, 501].includes(response.status())) {
      ledger.unexpected.push(`${response.request().method()} ${url.pathname}${url.search} ${response.status()}`);
    }
  };
  ledger.listeners.set(context, record);
  context.on("response", record);
}

export const test = base.extend<{ apiResponseLedger: void }>({
  apiResponseLedger: [async ({ context }, use, testInfo) => {
    const ledger: ApiResponseLedger = { unexpected: [], listeners: new Map() };
    ledgers.set(testInfo, ledger);
    registerMockContext(context);
    try {
      await use();
    } finally {
      for (const [registeredContext, listener] of ledger.listeners) registeredContext.off("response", listener);
      ledgers.delete(testInfo);
      assertNoUnexpectedApiResponses(ledger.unexpected);
    }
  }, { auto: true }],
});
export { expect };
