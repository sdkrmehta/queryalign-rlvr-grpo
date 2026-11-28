import React, { useState, useMemo } from 'react';
import {
  History,
  Search,
  Star,
  Trash2,
  Play,
  Copy,
  Check,
  Download,
  Clock,
  Database,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { QueryHistoryItem, InferenceMode } from '../types';
import {
  getStoredHistory,
  toggleFavoriteHistoryItem,
  deleteHistoryItem,
  clearHistory,
} from '../services/storage';

interface HistoryPageProps {
  onLoadQueryIntoWorkspace: (question: string, databaseId: string, mode: InferenceMode) => void;
  onRefreshHistoryBadge: () => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  onLoadQueryIntoWorkspace,
  onRefreshHistoryBadge,
}) => {
  const [historyItems, setHistoryItems] = useState<QueryHistoryItem[]>(() => getStoredHistory());
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDb, setSelectedDb] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [selectedMode, setSelectedMode] = useState<string>('all');
  const [onlyFavorites, setOnlyFavorites] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Distinct databases present in history
  const uniqueDatabases = useMemo(() => {
    const set = new Set<string>();
    historyItems.forEach(i => set.add(i.databaseId));
    return Array.from(set);
  }, [historyItems]);

  // Filtered Items
  const filteredItems = useMemo(() => {
    return historyItems.filter(item => {
      if (onlyFavorites && !item.isFavorite) return false;
      if (selectedDb !== 'all' && item.databaseId !== selectedDb) return false;
      if (selectedStatus !== 'all' && item.status !== selectedStatus) return false;
      if (selectedMode !== 'all' && item.mode !== selectedMode) return false;
      if (searchTerm.trim()) {
        const q = searchTerm.toLowerCase();
        const matchesQ = item.question.toLowerCase().includes(q);
        const matchesSql = item.sql ? item.sql.toLowerCase().includes(q) : false;
        if (!matchesQ && !matchesSql) return false;
      }
      return true;
    });
  }, [historyItems, onlyFavorites, selectedDb, selectedStatus, selectedMode, searchTerm]);

  const handleToggleFavorite = (id: string) => {
    const updated = toggleFavoriteHistoryItem(id);
    setHistoryItems(updated);
  };

  const handleDeleteItem = (id: string) => {
    const updated = deleteHistoryItem(id);
    setHistoryItems(updated);
    onRefreshHistoryBadge();
  };

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to delete all stored query history? This cannot be undone.')) {
      clearHistory();
      setHistoryItems([]);
      onRefreshHistoryBadge();
    }
  };

  const handleCopySql = (id: string, sql?: string) => {
    if (!sql) return;
    navigator.clipboard.writeText(sql);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleExportCsv = () => {
    if (filteredItems.length === 0) return;
    const header = ['Timestamp', 'Database', 'Mode', 'Status', 'Question', 'SQL', 'Rows', 'LatencyMs'];
    const rows = filteredItems.map(item => [
      new Date(item.timestamp).toISOString(),
      item.databaseId,
      item.mode,
      item.status,
      `"${item.question.replace(/"/g, '""')}"`,
      `"${(item.sql || '').replace(/"/g, '""')}"`,
      item.rowCount ?? '',
      item.latencyMs ?? '',
    ]);
    const csvContent = [header.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `queryalign_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header and Controls */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <History size={20} color="var(--qa-accent)" />
              <h2 style={{ fontSize: '18px', fontWeight: 800 }}>Query History & Telemetry Log</h2>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
              Review, re-run, or export previously executed natural language queries and generated SQL statements.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={handleExportCsv}
              disabled={filteredItems.length === 0}
              className="qa-btn-ghost"
              style={{ fontSize: '12px', gap: '6px' }}
            >
              <Download size={14} />
              <span>Export CSV</span>
            </button>
            <button
              onClick={handleClearAll}
              disabled={historyItems.length === 0}
              className="qa-btn-ghost"
              style={{ fontSize: '12px', gap: '6px', color: 'var(--qa-danger)' }}
            >
              <Trash2 size={14} />
              <span>Clear All</span>
            </button>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', paddingTop: '4px' }}>
          {/* Search Input */}
          <div style={{ position: 'relative', flex: '1 1 240px' }}>
            <Search
              size={14}
              style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--qa-text-muted)' }}
            />
            <input
              type="text"
              placeholder="Search questions or SQL..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="qa-input"
              style={{ paddingLeft: '32px', height: '34px', fontSize: '12px' }}
            />
          </div>

          {/* Database Filter */}
          <select
            value={selectedDb}
            onChange={e => setSelectedDb(e.target.value)}
            className="qa-select"
            style={{ width: '160px', height: '34px', fontSize: '12px' }}
          >
            <option value="all">All Databases</option>
            {uniqueDatabases.map(db => (
              <option key={db} value={db}>
                {db}
              </option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={e => setSelectedStatus(e.target.value)}
            className="qa-select"
            style={{ width: '140px', height: '34px', fontSize: '12px' }}
          >
            <option value="all">All Statuses</option>
            <option value="success">Success</option>
            <option value="format_error">Format Error</option>
            <option value="ast_rejected">AST Rejected</option>
            <option value="error">Error</option>
          </select>

          {/* Mode Filter */}
          <select
            value={selectedMode}
            onChange={e => setSelectedMode(e.target.value)}
            className="qa-select"
            style={{ width: '150px', height: '34px', fontSize: '12px' }}
          >
            <option value="all">All Modes</option>
            <option value="greedy">Greedy (Pass@1)</option>
            <option value="majority8">Majority@8</option>
          </select>

          {/* Favorites Filter */}
          <button
            onClick={() => setOnlyFavorites(!onlyFavorites)}
            className="qa-btn-ghost"
            style={{
              height: '34px',
              fontSize: '12px',
              backgroundColor: onlyFavorites ? 'rgba(234, 179, 8, 0.15)' : 'transparent',
              color: onlyFavorites ? '#EAB308' : 'var(--qa-text-muted)',
              border: onlyFavorites ? '1px solid rgba(234, 179, 8, 0.4)' : '1px solid var(--qa-border)',
            }}
          >
            <Star size={14} fill={onlyFavorites ? '#EAB308' : 'none'} />
            <span>Favorites</span>
          </button>
        </div>
      </div>

      {/* History Items List */}
      {filteredItems.length === 0 ? (
        <div
          style={{
            padding: '3rem 1rem',
            textAlign: 'center',
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            color: 'var(--qa-text-muted)',
          }}
        >
          <History size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
          <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>
            No query history found
          </div>
          <div style={{ fontSize: '13px', marginTop: '4px' }}>
            {historyItems.length === 0
              ? 'Execute questions in the Query Workspace to build up your execution log.'
              : 'No queries match your current filter settings.'}
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {filteredItems.map(item => {
            const isSuccess = item.status === 'success';
            const isCopied = copiedId === item.id;

            return (
              <div
                key={item.id}
                style={{
                  backgroundColor: 'var(--qa-bg-card)',
                  border: '1px solid var(--qa-border)',
                  borderRadius: 'var(--qa-radius-md)',
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                  transition: 'border-color 0.15s ease',
                }}
              >
                {/* Meta Row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    {/* Status Badge */}
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11px',
                        fontWeight: 700,
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: isSuccess ? 'var(--qa-success-bg)' : 'var(--qa-danger-bg)',
                        color: isSuccess ? 'var(--qa-success)' : 'var(--qa-danger)',
                        border: `1px solid ${isSuccess ? 'var(--qa-success-border)' : 'var(--qa-danger-border)'}`,
                        textTransform: 'uppercase',
                      }}
                    >
                      {isSuccess ? <CheckCircle size={12} /> : <XCircle size={12} />}
                      {item.status.replace('_', ' ')}
                    </span>

                    {/* Database Badge */}
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11px',
                        fontFamily: 'var(--qa-font-mono)',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: 'var(--qa-bg-elevated)',
                        color: 'var(--qa-text-secondary)',
                        border: '1px solid var(--qa-border)',
                      }}
                    >
                      <Database size={12} />
                      {item.databaseId}
                    </span>

                    {/* Mode Badge */}
                    <span
                      style={{
                        fontSize: '11px',
                        fontFamily: 'var(--qa-font-mono)',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: 'var(--qa-accent-subtle)',
                        color: 'var(--qa-accent)',
                        border: '1px solid rgba(129, 150, 255, 0.25)',
                      }}
                    >
                      {item.mode === 'majority8' ? 'Majority@8' : 'Greedy'}
                    </span>

                    {/* Timestamp */}
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '11px',
                        color: 'var(--qa-text-muted)',
                      }}
                    >
                      <Clock size={12} />
                      {new Date(item.timestamp).toLocaleString()}
                    </span>

                    {/* Row Count / Latency */}
                    {item.rowCount !== undefined && (
                      <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)', fontFamily: 'var(--qa-font-mono)' }}>
                        • {item.rowCount} row(s)
                      </span>
                    )}
                    {item.latencyMs !== undefined && (
                      <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)', fontFamily: 'var(--qa-font-mono)' }}>
                        • {item.latencyMs.toFixed(1)} ms
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <button
                      onClick={() => handleToggleFavorite(item.id)}
                      className="qa-btn-ghost"
                      style={{ padding: '4px', color: item.isFavorite ? '#EAB308' : 'var(--qa-text-muted)' }}
                      title={item.isFavorite ? 'Remove Favorite' : 'Mark as Favorite'}
                    >
                      <Star size={15} fill={item.isFavorite ? '#EAB308' : 'none'} />
                    </button>
                    <button
                      onClick={() => handleCopySql(item.id, item.sql)}
                      disabled={!item.sql}
                      className="qa-btn-ghost"
                      style={{ padding: '4px' }}
                      title="Copy SQL to Clipboard"
                    >
                      {isCopied ? <Check size={15} color="var(--qa-success)" /> : <Copy size={15} />}
                    </button>
                    <button
                      onClick={() => onLoadQueryIntoWorkspace(item.question, item.databaseId, item.mode)}
                      className="qa-btn-primary"
                      style={{ padding: '4px 10px', fontSize: '11px', gap: '4px' }}
                      title="Load into Query Workspace"
                    >
                      <Play size={12} fill="currentColor" />
                      <span>Run in Workspace</span>
                    </button>
                    <button
                      onClick={() => handleDeleteItem(item.id)}
                      className="qa-btn-ghost"
                      style={{ padding: '4px', color: 'var(--qa-danger)' }}
                      title="Delete from History"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </div>

                {/* Natural Language Question */}
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>
                  {item.question}
                </div>

                {/* SQL Code Block */}
                {item.sql && (
                  <pre
                    style={{
                      margin: 0,
                      padding: '10px 12px',
                      backgroundColor: 'var(--qa-bg-primary)',
                      border: '1px solid var(--qa-border)',
                      borderRadius: 'var(--qa-radius-sm)',
                      fontSize: '12px',
                      fontFamily: 'var(--qa-font-mono)',
                      color: 'var(--qa-accent)',
                      overflowX: 'auto',
                    }}
                  >
                    <code>{item.sql}</code>
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
