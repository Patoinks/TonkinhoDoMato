// Shared by the independently published ANW applications. Use a document link
// so their React routers cannot turn the standalone tool into a fallback route.
(() => {
  const href = '/ai-tier-maker/';
  const selectors = '.navbar-links, .mobile-nav-links, .anw-site-links, .anw-site-menu > div';
  function connect() {
    document.querySelectorAll(selectors).forEach(menu => {
      if (menu.querySelector('a[href="' + href + '"]')) return;
      const link = document.createElement('a');
      link.href = href;
      link.textContent = 'AI TIER MAKER';
      menu.append(link);
    });
  }
  // The existing desktop menu is crowded at tablet widths after adding a tab.
  const style = document.createElement('style');
  style.textContent = '@media(min-width:769px) and (max-width:1200px){.navbar{padding-left:24px;padding-right:24px}.navbar .left-items{gap:24px}.navbar-links{gap:18px}.navbar-links a,.nav-dropdown-trigger{font-size:12px;letter-spacing:.5px}}';
  document.head.append(style);
  connect();
  new MutationObserver(connect).observe(document.body, { childList: true, subtree: true });
})();
