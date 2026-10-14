export const BOT_BASE_NAME = 'Tonkinho';
export const BOT_VERSION = 'V2';
export const BOT_NAME = BOT_BASE_NAME;
export const BOT_AVATAR = '/tonkai/tonkinho-do-mato.png';
// Earlier names stay recognised so saved wins keep their side.
const EARLIER_BASE_NAMES = ['tonkinho filho', 'tonkclaudinho filhodegei'];
export const isBotName = (name?: string) => {
  const value = name?.trim().toLowerCase() ?? '';
  if (value === 'g16 v2' || value === 'g4' || value === 'g4 leto') return true;
  return value === 'tonkai' || value === 'tonkinho do mato' || /^tonkinho do mato v\d+$/.test(value)
    || [BOT_BASE_NAME.toLowerCase(), ...EARLIER_BASE_NAMES].some(base => value === base || value.startsWith(base + ' '));
};
