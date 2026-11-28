import React from 'react';
import { Moon, Sun, ShieldCheck, Activity, Terminal } from 'lucide-react';
import { ActivePage } from '../../types';

interface HeaderProps {
  activePage: ActivePage;
  theme: 'dark' | 'light';
  setTheme: (t: 'dark' | 'light') => void;
  isBackendOnline: boolean;
  selectedDatabase: string;
}

export const Header: React.FC<HeaderProps> = ({
  activePage,
  theme,
  setTheme,
  isBackendOnline,
  selectedDatabase,
}) => {
  return (
    <header
      style={{
        height: 'var(--qa-header-height)',
        backgroundColor: 'var(--qa-bg-card)',
        borderBottom: '1px solid var(--qa-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.5rem',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Left: Active Section & Context */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '14px',
              fontWeight: 700,
              textTransform: 'capitalize',
              color: 'var(--qa-text-primary)',
            }}
          >
            {activePage.replace('-', ' ')}
          </span>
          <span style={{ color: 'var(--qa-text-muted)' }}>/</span>
          <span
            style={{
              fontSize: '12px',
              fontFamily: 'var(--qa-font-mono)',
              color: 'var(--qa-accent)',
              backgroundColor: 'var(--qa-accent-subtle)',
              padding: '2px 8px',
              borderRadius: '4px',
              border: '1px solid rgba(129, 150, 255, 0.25)',
            }}
          >
            {selectedDatabase}.sqlite
          </span>
        </div>
      </div>

      {/* Right: Telemetry & Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Sandbox Security Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--qa-text-secondary)',
            backgroundColor: 'var(--qa-bg-elevated)',
            border: '1px solid var(--qa-border)',
            padding: '4px 10px',
            borderRadius: 'var(--qa-radius-sm)',
          }}
        >
          <ShieldCheck size={14} color="var(--qa-success)" />
          <span>RO Sandbox Enforced</span>
        </div>

        {/* Backend Online/Offline Status Indicator */}
        <div
          title={isBackendOnline ? 'FastAPI serving backend is connected and healthy.' : 'Backend unreachable. Run: uvicorn sql_grpo.serving.api:app --port 8000'}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            fontWeight: 600,
            padding: '4px 10px',
            borderRadius: 'var(--qa-radius-sm)',
            border: `1px solid ${isBackendOnline ? 'var(--qa-success-border)' : 'var(--qa-warning-border)'}`,
            backgroundColor: isBackendOnline ? 'var(--qa-success-bg)' : 'var(--qa-warning-bg)',
            color: isBackendOnline ? 'var(--qa-success)' : 'var(--qa-warning)',
          }}
        >
          <Activity size={14} />
          <span>{isBackendOnline ? 'API Connected' : 'API Offline (Port 8000)'}</span>
        </div>

        {/* Dialect Info */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '11px',
            color: 'var(--qa-text-muted)',
            fontFamily: 'var(--qa-font-mono)',
          }}
        >
          <Terminal size={12} />
          <span>SQLite 3.x</span>
        </div>

        {/* Theme Switcher Toggle */}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="qa-btn-ghost"
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Theme`}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: 'var(--qa-radius-sm)',
            padding: 0,
            justifyContent: 'center',
          }}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
};
