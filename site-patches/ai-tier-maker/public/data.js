// Top 20 do leaderboard geral de texto da LMArena, transcrito da captura de 2026-10-16.
// score = "Net Improvement" (%), ci = intervalo (±%).
window.TIER_PROVIDERS = {
  anthropic: { name: 'Anthropic', color: '#d97757' },
  openai:    { name: 'OpenAI',    color: '#10a37f' },
  google:    { name: 'Google',    color: '#4285f4' },
  meta:      { name: 'Meta',      color: '#0a7cff' },
  xiaomi:    { name: 'Xiaomi',    color: '#ff6900' },
  xai:       { name: 'SpaceXAI',  color: '#cfd3dc' },
  deepseek:  { name: 'DeepSeek',  color: '#4d6bfe' },
};

window.TIER_MODELS = [
  { rank: 1,  name: 'Claude Opus 5.5',     effort: 'Max',   provider: 'anthropic', score: 17.18, ci: 3.38, license: 'Proprietary' },
  { rank: 2,  name: 'GPT 6.1 Sol',         effort: 'Max',   provider: 'openai',    score: 14.78, ci: 2.03, license: 'Proprietary' },
  { rank: 3,  name: 'Claude Fable 5.1',    effort: 'Max',   provider: 'anthropic', score: 14.12, ci: 2.01, license: 'Proprietary' },
  { rank: 4,  name: 'Gemini 4 Argon',      effort: 'High',  provider: 'google',    score: 13.73, ci: 1.52, license: 'Proprietary' },
  { rank: 5,  name: 'GPT 6 Astra',         effort: 'Max',   provider: 'openai',    score: 13.69, ci: 2.32, license: 'Proprietary' },
  { rank: 6,  name: 'Claude Sonnet 5.5',   effort: 'Max',   provider: 'anthropic', score: 12.55, ci: 3.13, license: 'Proprietary' },
  { rank: 7,  name: 'GPT 6 Sol',           effort: 'Max',   provider: 'openai',    score: 11.20, ci: 2.27, license: 'Proprietary' },
  { rank: 8,  name: 'Claude Opus 5',       effort: 'Max',   provider: 'anthropic', score: 9.20,  ci: 1.69, license: 'Proprietary' },
  { rank: 9,  name: 'Claude Fable 5',      effort: 'High',  provider: 'anthropic', score: 8.39,  ci: 1.41, license: 'Proprietary' },
  { rank: 10, name: 'GPT 5.6 Sol',         effort: 'xHigh', provider: 'openai',    score: 7.42,  ci: 1.41, license: 'Proprietary' },
  { rank: 11, name: 'Claude Opus 4.8',     effort: 'High',  provider: 'anthropic', score: 7.14,  ci: 1.53, license: 'Proprietary' },
  { rank: 12, name: 'Muse Wonder',         effort: 'Max',   provider: 'meta',      score: 6.42,  ci: 2.15, license: 'Proprietary' },
  { rank: 13, name: 'Claude Sonnet 5',     effort: 'High',  provider: 'anthropic', score: 4.69,  ci: 1.70, license: 'Proprietary' },
  { rank: 14, name: 'Gemini 3.8 Flash',    effort: 'High',  provider: 'google',    score: 3.97,  ci: 1.08, license: 'Proprietary' },
  { rank: 15, name: 'MiMo V2.6 Pro',       effort: '',      provider: 'xiaomi',    score: 3.81,  ci: 1.18, license: 'MIT' },
  { rank: 16, name: 'Muse Spark 1.3',      effort: 'Max',   provider: 'meta',      score: 3.63,  ci: 0.81, license: 'Proprietary' },
  { rank: 17, name: 'Grok 4.7',            effort: 'xHigh', provider: 'xai',       score: 3.39,  ci: 1.67, license: 'Proprietary' },
  { rank: 18, name: 'GPT 5.5',             effort: 'xHigh', provider: 'openai',    score: 3.31,  ci: 1.28, license: 'Proprietary' },
  { rank: 19, name: 'GPT 6 Luna',          effort: 'Max',   provider: 'openai',    score: 3.26,  ci: 0.49, license: 'Proprietary' },
  { rank: 20, name: 'Deepseek V4.1 Flash', effort: 'Max',   provider: 'deepseek',  score: 2.95,  ci: 0.37, license: 'MIT' },
].map(m => ({ ...m, id: 'm' + m.rank }));
