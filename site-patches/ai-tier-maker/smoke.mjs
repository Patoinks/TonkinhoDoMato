import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const require = createRequire('C:/Users/Martim/Desktop/Github/ANW-TONKAI-RELEASE-20260922/package.json');
const { chromium } = require('@playwright/test');
const here = dirname(fileURLToPath(import.meta.url)), out = join(here, 'evidence');
const live = process.argv.includes('--live');
const origin = 'https://anw-games.web.app';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  if (!live) await context.route(origin + '/**', async route => {
    const u = new URL(route.request().url());
    let path = u.pathname;
    if (['/', '/portfolio'].includes(path)) path = '/index.html';
    if (/^\/(tonkai|lol|ai-tier-maker)\/?$/.test(path)) path = path.replace(/\/$/, '') + '/index.html';
    const file = join(out, 'staged', path);
    if (!existsSync(file)) return route.continue();
    const ext = path.split('.').at(-1);
    await route.fulfill({ status: 200, body: readFileSync(file), contentType: ({ html: 'text/html', js: 'application/javascript', css: 'text/css', svg: 'image/svg+xml' })[ext] ?? 'application/octet-stream' });
  });
  const page = await context.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  // Enter the tool from each independently hosted app, using the actual menu.
  for (const path of ['/', '/lol/', '/tonkai']) {
    await page.goto(origin + path);
    const link = page.locator('a[href="/ai-tier-maker/"]:visible').first();
    await link.waitFor(); await link.click();
    await page.getByRole('heading', { name: 'AI Tier Maker', exact: true }).waitFor();
    await page.locator('.card').first().waitFor();
    assert.equal(new URL(page.url()).pathname, '/ai-tier-maker/');
    assert.equal(await page.locator('.card').count(), 20);
  }
  // Both URL forms load CSS/JS using the same base path.
  await page.goto(origin + '/ai-tier-maker');
  await page.locator('.card').first().waitFor();
  await page.locator('#btn-reset').click();
  assert.equal(await page.locator('#pool .card').count(), 20);
  assert.equal(await page.locator('.tier').count(), 6);
  await page.locator('.card img').evaluateAll(images => Promise.all(images.map(i => i.decode())));
  assert.ok(await page.locator('.card img').evaluateAll(images => images.every(i => i.complete && i.naturalWidth > 0)));
  await page.locator('#pool .card').first().click();
  await page.keyboard.press('s');
  assert.equal(await page.locator('.tier').first().locator('.card').count(), 1);
  await page.reload();
  await page.locator('.tier .card').first().waitFor();
  assert.equal(await page.locator('.tier').first().locator('.card').count(), 1);
  // Pointer drag between pool and a tier, then auto-sort all models.
  await page.setViewportSize({ width: 1440, height: 2200 });
  const from = await page.locator('#pool .card').first().boundingBox();
  const to = await page.locator('.tier .dropzone').nth(1).boundingBox();
  await page.mouse.move(from.x + 40, from.y + 35); await page.mouse.down();
  await page.mouse.move(to.x + 50, to.y + 45, { steps: 16 }); await page.mouse.up();
  assert.equal(await page.locator('.tier').nth(1).locator('.card').count(), 1);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('#btn-auto').click();
  assert.equal(await page.locator('#board .card').count(), 20);
  assert.equal(await page.locator('#pool .card').count(), 0);
  await page.locator('label[for="toggle-rank"]').click();
  assert.ok(await page.locator('body').evaluate(el => el.classList.contains('hide-rank')));
  await page.locator('label[for="toggle-rank"]').click();
  await page.locator('#btn-add-tier').click();
  assert.equal(await page.locator('.tier').count(), 7);
  await page.locator('.tier').last().getByRole('button', { name: 'Apagar tier' }).click();
  assert.equal(await page.locator('.tier').count(), 6);
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#btn-export').click();
  const download = await downloadEvent;
  const pngPath = join(out, (live ? 'live' : 'preview') + '-export.png');
  await download.saveAs(pngPath);
  const png = readFileSync(pngPath);
  assert.equal(png.subarray(1, 4).toString(), 'PNG');
  assert.equal(png.readUInt32BE(16), 3200);
  await page.evaluate(() => { scrollTo(0, 0); return document.fonts.ready; });
  await page.screenshot({ path: join(out, (live ? 'live' : 'preview') + '-desktop.png'), fullPage: true, animations: 'disabled' });
  const layouts = [];
  for (const width of [1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Tool overflow at ' + width);
    assert.ok(await page.locator('#btn-export').isVisible()); layouts.push(width);
  }
  await page.setViewportSize({ width: 390, height: 900 });
  await page.reload();
  await page.locator('.tier .card').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: join(out, (live ? 'live' : 'preview') + '-mobile.png'), fullPage: true, animations: 'disabled' });
  await page.locator('.tier').nth(3).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(out, (live ? 'live' : 'preview') + '-mobile-lower.png'), animations: 'disabled' });
  // Return links use document navigation back to the original apps.
  await page.locator('.site-links a[href="/lol/"]').click();
  await page.locator('.anw-site-menu summary').click();
  await page.locator('.anw-site-menu a[href="/ai-tier-maker/"]').click();
  await page.getByRole('heading', { name: 'AI Tier Maker', exact: true }).waitFor();
  await page.locator('.site-links a[href="/"]').click();
  await page.locator('.hamburger-menu').click();
  await page.locator('.mobile-nav-links a[href="/ai-tier-maker/"]').click();
  await page.getByRole('heading', { name: 'AI Tier Maker', exact: true }).waitFor();
  for (const path of ['/', '/lol/', '/tonkai']) {
    await page.goto(origin + path);
    await page.locator('a[href="/ai-tier-maker/"]').first().waitFor({ state: 'attached' });
    for (const width of [1440, 1024, 820, 769, 390]) {
      await page.setViewportSize({ width, height: 900 });
      const fits = await page.locator('nav a:visible').evaluateAll(links => links.every(a => { const r = a.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth; }));
      assert.ok(fits, path + ' navbar clips links at ' + width);
    }
  }
  assert.deepEqual(errors, []);
  const report = { live, navigationFromAllApps: true, mobileMenus: true, cards: 20, dragAndKeyboard: true, persistence: true, autoSort: true, exportPng: { width: png.readUInt32BE(16), height: png.readUInt32BE(20) }, responsiveWidths: layouts, errors };
  writeFileSync(join(out, (live ? 'live' : 'preview') + '-smoke.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await browser.close(); }
