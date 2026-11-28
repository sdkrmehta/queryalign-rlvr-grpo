import React, { useState } from 'react';
import {
  Settings,
  Moon,
  Sun,
  Shield,
  Trash2,
  Activity,
  Zap,
  Check,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { InferenceMode } from '../types';
import {
  clearHistory,
  setStoredMode,
  setStoredTheme,
} from '../services/storage';
import { checkBackendHealth } from '../services/api';

interface SettingsPageProps {
  theme: 'dark' | 'light';
  setTheme: (t: 'dark' | 'light') => void;
  defaultMode: InferenceMode;
  setDefaultMode: (m: InferenceMode) => void;
  onRefreshHistoryBadge: () => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  theme,
  setTheme,
  defaultMode,
  setDefaultMode,
  onRefreshHistoryBadge,
}) => {
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [healthStatus, setHealthStatus] = useState<{ status: string; service: string } | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState(false);

  const handleModeChange = (mode: InferenceMode) => {
    setDefaultMode(mode);
    setStoredMode(mode);
    showNotice();
  };

  const handleThemeChange = (newTheme: 'dark' | 'light') => {
    setTheme(newTheme);
    setStoredTheme(newTheme);
    showNotice();
  };

  const showNotice = () => {
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 2000);
  };

  const handleHealthCheck = async () => {
    setIsCheckingHealth(true);
    setHealthStatus(null);
    setHealthError(null);
    try {
      const res = await checkBackendHealth();
      setHealthStatus(res);
    } catch (err: unknown) {
      setHealthError(err instanceof Error ? err.message : 'Backend unreachable.');
    } finally {
      setIsCheckingHealth(false);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to delete all stored query history?')) {
      clearHistory();
      onRefreshHistoryBadge();
      alert('Query history cleared.');
    }
  };

  return (
    <div style={{ maxWidth: '960px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header Banner */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                backgroundColor: 'var(--qa-accent-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--qa-accent)',
                border: '1px solid rgba(129, 150, 255, 0.3)',
              }}
            >
              <Settings size={20} />
            </div>
            <div>
              <h1 style={{ fontSize: '18px', fontWeight: 800 }}>System Settings & Preferences</h1>
              <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Configure frontend behavior, default inference algorithms, and verify backend connection health.
              </p>
            </div>
          </div>

          {savedNotice && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--qa-success)',
                backgroundColor: 'var(--qa-success-bg)',
                padding: '4px 10px',
                borderRadius: 'var(--qa-radius-sm)',
                border: '1px solid var(--qa-success-border)',
              }}
            >
              <Check size={14} />
              <span>Preference Saved</span>
            </div>
          )}
        </div>
      </div>

      {/* Preferences Sections */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        {/* Appearance Section */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
          }}
        >
          <h2 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>Appearance & Theme</h2>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>Interface Theme</div>
              <div style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Toggle between the dark engineering workbench palette and the high-contrast light theme.
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => handleThemeChange('dark')}
                className="qa-btn-ghost"
                style={{
                  backgroundColor: theme === 'dark' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: theme === 'dark' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                  border: theme === 'dark' ? '1px solid var(--qa-accent)' : '1px solid var(--qa-border)',
                }}
              >
                <Moon size={15} />
                <span>Dark Theme</span>
              </button>
              <button
                onClick={() => handleThemeChange('light')}
                className="qa-btn-ghost"
                style={{
                  backgroundColor: theme === 'light' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: theme === 'light' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                  border: theme === 'light' ? '1px solid var(--qa-accent)' : '1px solid var(--qa-border)',
                }}
              >
                <Sun size={15} />
                <span>Light Theme</span>
              </button>
            </div>
          </div>
        </div>

        {/* Inference Mode Section */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
          }}
        >
          <h2 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>Default Inference Mode</h2>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>Default Strategy</div>
              <div style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Choose default mode for new queries: Greedy Pass@1 (fast single-shot) or Majority@8 (self-consistency search).
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => handleModeChange('greedy')}
                className="qa-btn-ghost"
                style={{
                  backgroundColor: defaultMode === 'greedy' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: defaultMode === 'greedy' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                  border: defaultMode === 'greedy' ? '1px solid var(--qa-accent)' : '1px solid var(--qa-border)',
                }}
              >
                <Zap size={14} />
                <span>Greedy (Pass@1)</span>
              </button>
              <button
                onClick={() => handleModeChange('majority8')}
                className="qa-btn-ghost"
                style={{
                  backgroundColor: defaultMode === 'majority8' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: defaultMode === 'majority8' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                  border: defaultMode === 'majority8' ? '1px solid var(--qa-accent)' : '1px solid var(--qa-border)',
                }}
              >
                <Activity size={14} />
                <span>Majority@8</span>
              </button>
            </div>
          </div>
        </div>

        {/* Backend Endpoint & Health Check */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
          }}
        >
          <h2 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>Backend Serving Connection</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>FastAPI Proxy Route</div>
                <div style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                  Vite proxy routes API calls to <code>http://127.0.0.1:8000</code>.
                </div>
              </div>

              <button
                onClick={handleHealthCheck}
                disabled={isCheckingHealth}
                className="qa-btn-ghost"
                style={{ gap: '6px' }}
              >
                <RefreshCw size={14} className={isCheckingHealth ? 'qa-spin' : ''} />
                <span>Ping Health Endpoint</span>
              </button>
            </div>

            {healthStatus && (
              <div
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--qa-radius-sm)',
                  backgroundColor: 'var(--qa-success-bg)',
                  border: '1px solid var(--qa-success-border)',
                  color: 'var(--qa-success)',
                  fontSize: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Activity size={14} />
                <span>
                  Backend Online: Status = <strong>{healthStatus.status}</strong>, Service = <strong>{healthStatus.service}</strong>
                </span>
              </div>
            )}

            {healthError && (
              <div
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--qa-radius-sm)',
                  backgroundColor: 'var(--qa-danger-bg)',
                  border: '1px solid var(--qa-danger-border)',
                  color: 'var(--qa-danger)',
                  fontSize: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <AlertCircle size={14} />
                <span>{healthError}</span>
              </div>
            )}
          </div>
        </div>

        {/* Data & History Storage */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
          }}
        >
          <h2 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>Local Storage & History</h2>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>Purge Saved History</div>
              <div style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Delete all cached queries, SQL responses, and telemetry from your browser's local storage.
              </div>
            </div>

            <button
              onClick={handleClearHistory}
              className="qa-btn-ghost"
              style={{ color: 'var(--qa-danger)', gap: '6px' }}
            >
              <Trash2 size={14} />
              <span>Clear History</span>
            </button>
          </div>
        </div>

        {/* Security Policy Information */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Shield size={16} color="var(--qa-success)" />
            <h2 style={{ fontSize: '15px', fontWeight: 700 }}>Active Sandbox Safeguards (Immutable)</h2>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)', marginBottom: '12px' }}>
            These sandbox policies are enforced strictly at the C and SQLite virtual machine layer in the backend:
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '8px', fontSize: '12px' }}>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <strong>URI Mode:</strong> <code>file:?mode=ro</code>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <strong>PRAGMA:</strong> <code>query_only = ON</code>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <strong>Authorizer:</strong> <code>C callback hook</code>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <strong>Deadline:</strong> <code>2.0s watchdog limit</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
