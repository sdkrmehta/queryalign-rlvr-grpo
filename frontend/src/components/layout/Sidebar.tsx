import React from 'react';
import {
  Sparkles,
  History,
  BarChart3,
  Cpu,
  Database,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
} from 'lucide-react';
import { ActivePage } from '../../types';

interface SidebarProps {
  activePage: ActivePage;
  setActivePage: (p: ActivePage) => void;
  collapsed: boolean;
  setCollapsed: (c: boolean) => void;
  historyCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  setActivePage,
  collapsed,
  setCollapsed,
  historyCount,
}) => {
  const navItems: { id: ActivePage; label: string; icon: React.ReactNode; badge?: number }[] = [
    { id: 'workspace', label: 'Query Workspace', icon: <Sparkles size={17} /> },
    { id: 'history', label: 'Query History', icon: <History size={17} />, badge: historyCount },
    { id: 'evaluation', label: 'Evaluation Dashboard', icon: <BarChart3 size={17} /> },
    { id: 'model-info', label: 'Model Information', icon: <Cpu size={17} /> },
    { id: 'schema-explorer', label: 'Schema Explorer', icon: <Database size={17} /> },
    { id: 'settings', label: 'Settings', icon: <Settings size={17} /> },
  ];

  return (
    <aside
      style={{
        width: collapsed ? 'var(--qa-sidebar-collapsed-width)' : 'var(--qa-sidebar-width)',
        backgroundColor: 'var(--qa-bg-sidebar)',
        borderRight: '1px solid var(--qa-border)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        transition: 'width 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        position: 'sticky',
        top: 0,
        height: '100vh',
        zIndex: 60,
        flexShrink: 0,
      }}
    >
      {/* Top Branding Section */}
      <div>
        <div
          style={{
            height: 'var(--qa-header-height)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'space-between',
            padding: collapsed ? '0' : '0 1rem',
            borderBottom: '1px solid var(--qa-border)',
          }}
        >
          {!collapsed ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #8196FF 0%, #465DC8 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#FFFFFF',
                  fontWeight: 800,
                  fontSize: '16px',
                  boxShadow: '0 2px 8px rgba(129, 150, 255, 0.3)',
                }}
              >
                ⚡
              </div>
              <div>
                <div style={{ fontWeight: 800, fontSize: '16px', lineHeight: 1.1 }}>QueryAlign</div>
                <div style={{ fontSize: '10px', color: 'var(--qa-text-muted)', fontWeight: 600 }}>
                  RLVR + GRPO SQL AI
                </div>
              </div>
            </div>
          ) : (
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #8196FF 0%, #465DC8 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#FFFFFF',
                fontWeight: 800,
                fontSize: '16px',
              }}
            >
              ⚡
            </div>
          )}

          <button
            onClick={() => setCollapsed(!collapsed)}
            className="qa-btn-ghost"
            style={{ width: '28px', height: '28px', padding: 0, justifyContent: 'center' }}
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* Navigation List */}
        <nav style={{ padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map(item => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                title={collapsed ? item.label : undefined}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  gap: '10px',
                  padding: collapsed ? '10px 0' : '9px 12px',
                  borderRadius: 'var(--qa-radius-sm)',
                  backgroundColor: isActive ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: isActive ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
                  fontWeight: isActive ? 600 : 500,
                  fontSize: '13px',
                  border: isActive ? '1px solid rgba(129, 150, 255, 0.25)' : '1px solid transparent',
                  position: 'relative',
                }}
              >
                <span>{item.icon}</span>
                {!collapsed && <span style={{ flex: 1, textAlign: 'left' }}>{item.label}</span>}
                {!collapsed && item.badge !== undefined && item.badge > 0 && (
                  <span
                    style={{
                      fontSize: '10px',
                      fontFamily: 'var(--qa-font-mono)',
                      backgroundColor: 'var(--qa-bg-elevated)',
                      color: 'var(--qa-text-muted)',
                      padding: '1px 6px',
                      borderRadius: '10px',
                      border: '1px solid var(--qa-border)',
                    }}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Telemetry Card (when expanded) */}
      {!collapsed && (
        <div style={{ padding: '12px', margin: '8px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-border)', borderRadius: 'var(--qa-radius-md)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
            <Shield size={13} color="var(--qa-success)" />
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-success)', textTransform: 'uppercase' }}>
              Sandbox Enforced
            </span>
          </div>
          <div style={{ fontSize: '10px', color: 'var(--qa-text-muted)', display: 'flex', flexDirection: 'column', gap: '2px', fontFamily: 'var(--qa-font-mono)' }}>
            <div>&bull; URI: file:?mode=ro</div>
            <div>&bull; PRAGMA: query_only=ON</div>
            <div>&bull; SQLite C Authorizer Active</div>
            <div>&bull; VM Progress: 2.0s limit</div>
          </div>
        </div>
      )}
    </aside>
  );
};
