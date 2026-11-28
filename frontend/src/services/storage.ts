import { InferenceMode, QueryHistoryItem } from '../types';

const STORAGE_KEYS = {
  HISTORY: 'queryalign_query_history_v1',
  THEME: 'queryalign_theme_preference',
  DEFAULT_MODE: 'queryalign_default_mode',
  AUTO_EXECUTE: 'queryalign_auto_execute',
};

export function getStoredHistory(): QueryHistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.HISTORY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (err) {
    console.error('Failed to read history from localStorage:', err);
    return [];
  }
}

export function saveHistoryItem(item: Omit<QueryHistoryItem, 'id' | 'timestamp'>): QueryHistoryItem {
  const history = getStoredHistory();
  const newItem: QueryHistoryItem = {
    ...item,
    id: `hist_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    timestamp: Date.now(),
  };

  const updated = [newItem, ...history.filter(h => h.question !== item.question || h.databaseId !== item.databaseId)].slice(0, 100);
  try {
    localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(updated));
  } catch (err) {
    console.error('Failed to save history to localStorage:', err);
  }
  return newItem;
}

export function toggleFavoriteHistoryItem(id: string): QueryHistoryItem[] {
  const history = getStoredHistory();
  const updated = history.map(item =>
    item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
  );
  localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(updated));
  return updated;
}

export function deleteHistoryItem(id: string): QueryHistoryItem[] {
  const history = getStoredHistory();
  const updated = history.filter(item => item.id !== id);
  localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(updated));
  return updated;
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEYS.HISTORY);
}

export function getStoredTheme(): 'dark' | 'light' {
  const saved = localStorage.getItem(STORAGE_KEYS.THEME);
  if (saved === 'light' || saved === 'dark') return saved;
  return 'dark'; // Dark theme is default for QueryAlign
}

export function setStoredTheme(theme: 'dark' | 'light'): void {
  localStorage.setItem(STORAGE_KEYS.THEME, theme);
  document.documentElement.setAttribute('data-theme', theme);
}

export function getStoredMode(): InferenceMode {
  const saved = localStorage.getItem(STORAGE_KEYS.DEFAULT_MODE);
  if (saved === 'majority8') return 'majority8';
  return 'greedy';
}

export function setStoredMode(mode: InferenceMode): void {
  localStorage.setItem(STORAGE_KEYS.DEFAULT_MODE, mode);
}
