import React, { useState } from 'react';
import {
  Database,
  Search,
  Code2,
  Network,
  FileText,
  Key,
  Link2,
  ArrowRight,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { DatabaseMetadata } from '../types';
import { SchemaDiagram } from '../components/schema/SchemaDiagram';

interface SchemaExplorerPageProps {
  databases: string[];
  selectedDatabase: string;
  setSelectedDatabase: (db: string) => void;
  metadata: DatabaseMetadata | null;
  schemaPromptText: string;
  onNavigateToWorkspaceWithQuery: (initialQuestion: string, databaseId: string) => void;
}

export const SchemaExplorerPage: React.FC<SchemaExplorerPageProps> = ({
  databases,
  selectedDatabase,
  setSelectedDatabase,
  metadata,
  schemaPromptText,
  onNavigateToWorkspaceWithQuery,
}) => {
  const [activeTab, setActiveTab] = useState<'tables' | 'diagram' | 'prompt'>('tables');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedTables, setExpandedTables] = useState<Record<string, boolean>>({});

  const tables = metadata?.tables || {};
  const tableEntries = Object.entries(tables);
  const foreignKeys = metadata?.foreign_keys || [];

  const totalColumns = tableEntries.reduce((acc, [, t]) => acc + (t.columns?.length || 0), 0);

  const toggleTable = (name: string) => {
    setExpandedTables(prev => ({ ...prev, [name]: !prev[name] }));
  };

  const expandAll = () => {
    const all: Record<string, boolean> = {};
    tableEntries.forEach(([name]) => (all[name] = true));
    setExpandedTables(all);
  };

  const collapseAll = () => {
    setExpandedTables({});
  };

  const filteredTables = tableEntries.filter(([name, data]) => {
    if (!searchTerm.trim()) return true;
    const q = searchTerm.toLowerCase();
    if (name.toLowerCase().includes(q)) return true;
    return data.columns.some(col => col.name.toLowerCase().includes(q) || col.type.toLowerCase().includes(q));
  });

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header & Database Selection Toolbar */}
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
              <Database size={20} />
            </div>
            <div>
              <h1 style={{ fontSize: '18px', fontWeight: 800 }}>Database Schema Explorer</h1>
              <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Inspect table structures, primary keys, foreign key relations, and prompt encodings across active databases.
              </p>
            </div>
          </div>

          {/* Database Selector Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--qa-text-muted)' }}>Database:</span>
            <select
              value={selectedDatabase}
              onChange={e => setSelectedDatabase(e.target.value)}
              className="qa-select"
              style={{ minWidth: '180px', height: '36px' }}
            >
              {databases.map(db => (
                <option key={db} value={db}>
                  {db}.sqlite
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Telemetry Stats Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1.5rem',
            paddingTop: '8px',
            borderTop: '1px solid var(--qa-border)',
            fontSize: '12px',
            color: 'var(--qa-text-secondary)',
          }}
        >
          <div>
            <span style={{ color: 'var(--qa-text-muted)' }}>Tables: </span>
            <strong style={{ color: 'var(--qa-text-primary)' }}>{tableEntries.length}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--qa-text-muted)' }}>Columns: </span>
            <strong style={{ color: 'var(--qa-text-primary)' }}>{totalColumns}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--qa-text-muted)' }}>Foreign Keys: </span>
            <strong style={{ color: 'var(--qa-text-primary)' }}>{foreignKeys.length}</strong>
          </div>
        </div>
      </div>

      {/* Tabs and Content Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '10px',
        }}
      >
        {/* Navigation Tabs */}
        <div
          style={{
            display: 'flex',
            backgroundColor: 'var(--qa-bg-elevated)',
            padding: '4px',
            borderRadius: 'var(--qa-radius-sm)',
            border: '1px solid var(--qa-border)',
          }}
        >
          <button
            onClick={() => setActiveTab('tables')}
            className="qa-btn-ghost"
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              backgroundColor: activeTab === 'tables' ? 'var(--qa-bg-card)' : 'transparent',
              color: activeTab === 'tables' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
              boxShadow: activeTab === 'tables' ? 'var(--qa-shadow-sm)' : 'none',
            }}
          >
            <Code2 size={14} />
            <span>Structured Tables ({tableEntries.length})</span>
          </button>
          <button
            onClick={() => setActiveTab('diagram')}
            className="qa-btn-ghost"
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              backgroundColor: activeTab === 'diagram' ? 'var(--qa-bg-card)' : 'transparent',
              color: activeTab === 'diagram' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
              boxShadow: activeTab === 'diagram' ? 'var(--qa-shadow-sm)' : 'none',
            }}
          >
            <Network size={14} />
            <span>Entity Relationship Diagram</span>
          </button>
          <button
            onClick={() => setActiveTab('prompt')}
            className="qa-btn-ghost"
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              backgroundColor: activeTab === 'prompt' ? 'var(--qa-bg-card)' : 'transparent',
              color: activeTab === 'prompt' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
              boxShadow: activeTab === 'prompt' ? 'var(--qa-shadow-sm)' : 'none',
            }}
          >
            <FileText size={14} />
            <span>Prompt M-Schema</span>
          </button>
        </div>

        {/* Tab-specific Toolbar (for Tables view) */}
        {activeTab === 'tables' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ position: 'relative', width: '220px' }}>
              <Search
                size={13}
                style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--qa-text-muted)' }}
              />
              <input
                type="text"
                placeholder="Filter tables/columns..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="qa-input"
                style={{ paddingLeft: '30px', height: '32px', fontSize: '12px' }}
              />
            </div>
            <button onClick={expandAll} className="qa-btn-ghost" style={{ fontSize: '11px', padding: '4px 8px' }}>
              Expand All
            </button>
            <button onClick={collapseAll} className="qa-btn-ghost" style={{ fontSize: '11px', padding: '4px 8px' }}>
              Collapse All
            </button>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      {activeTab === 'tables' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {filteredTables.length === 0 ? (
            <div
              style={{
                padding: '2.5rem',
                textAlign: 'center',
                backgroundColor: 'var(--qa-bg-card)',
                border: '1px solid var(--qa-border)',
                borderRadius: 'var(--qa-radius-md)',
                color: 'var(--qa-text-muted)',
              }}
            >
              No tables or columns match "{searchTerm}".
            </div>
          ) : (
            filteredTables.map(([tableName, tableData]) => {
              const isExpanded = expandedTables[tableName] ?? true;
              const relatedFks = foreignKeys.filter(
                fk => fk.from_table === tableName || fk.to_table === tableName
              );

              return (
                <div
                  key={tableName}
                  style={{
                    backgroundColor: 'var(--qa-bg-card)',
                    border: '1px solid var(--qa-border)',
                    borderRadius: 'var(--qa-radius-md)',
                    overflow: 'hidden',
                  }}
                >
                  {/* Table Header Bar */}
                  <div
                    onClick={() => toggleTable(tableName)}
                    style={{
                      padding: '10px 14px',
                      backgroundColor: 'var(--qa-bg-elevated)',
                      borderBottom: isExpanded ? '1px solid var(--qa-border)' : 'none',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                      userSelect: 'none',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      <span style={{ fontWeight: 700, fontSize: '14px', color: 'var(--qa-text-primary)' }}>
                        {tableName}
                      </span>
                      <span
                        style={{
                          fontSize: '11px',
                          fontFamily: 'var(--qa-font-mono)',
                          color: 'var(--qa-text-muted)',
                          backgroundColor: 'var(--qa-bg-card)',
                          padding: '1px 6px',
                          borderRadius: '4px',
                          border: '1px solid var(--qa-border)',
                        }}
                      >
                        {tableData.columns.length} columns
                      </span>
                    </div>

                    <button
                      onClick={e => {
                        e.stopPropagation();
                        onNavigateToWorkspaceWithQuery(`List all ${tableName}`, selectedDatabase);
                      }}
                      className="qa-btn-ghost"
                      style={{ fontSize: '11px', gap: '4px', padding: '3px 8px' }}
                      title="Open query in Workspace"
                    >
                      <span>Query Table</span>
                      <ArrowRight size={12} />
                    </button>
                  </div>

                  {/* Columns List & Relations (when expanded) */}
                  {isExpanded && (
                    <div style={{ padding: '12px 16px' }}>
                      <table className="qa-table" style={{ width: '100%', marginBottom: relatedFks.length > 0 ? '12px' : 0 }}>
                        <thead>
                          <tr>
                            <th style={{ width: '220px' }}>Column Name</th>
                            <th style={{ width: '120px' }}>Data Type</th>
                            <th style={{ width: '160px' }}>Constraint</th>
                            <th>Sample Data Values</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tableData.columns.map(col => {
                            const isPk = col.is_pk;
                            const fk = foreignKeys.find(
                              f => f.from_table === tableName && f.from_column === col.name
                            );

                            return (
                              <tr key={col.name}>
                                <td style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)', fontSize: '12px' }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    {isPk && <span title="Primary Key" style={{ display: 'inline-flex' }}><Key size={13} color="#EAB308" /></span>}
                                    {fk && <span title="Foreign Key" style={{ display: 'inline-flex' }}><Link2 size={13} color="var(--qa-accent)" /></span>}
                                    <span>{col.name}</span>
                                  </div>
                                </td>
                                <td>
                                  <span
                                    style={{
                                      fontSize: '11px',
                                      fontFamily: 'var(--qa-font-mono)',
                                      color: 'var(--qa-text-muted)',
                                    }}
                                  >
                                    {col.type || 'TEXT'}
                                  </span>
                                </td>
                                <td>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    {isPk && (
                                      <span
                                        style={{
                                          fontSize: '10px',
                                          fontWeight: 700,
                                          padding: '1px 6px',
                                          borderRadius: '3px',
                                          backgroundColor: 'rgba(234, 179, 8, 0.15)',
                                          color: '#EAB308',
                                          border: '1px solid rgba(234, 179, 8, 0.3)',
                                        }}
                                      >
                                        PRIMARY KEY
                                      </span>
                                    )}
                                    {fk && (
                                      <span
                                        style={{
                                          fontSize: '10px',
                                          fontFamily: 'var(--qa-font-mono)',
                                          padding: '1px 6px',
                                          borderRadius: '3px',
                                          backgroundColor: 'var(--qa-accent-subtle)',
                                          color: 'var(--qa-accent)',
                                          border: '1px solid rgba(129, 150, 255, 0.25)',
                                        }}
                                      >
                                        &rarr; {fk.to_table}.{fk.to_column}
                                      </span>
                                    )}
                                    {!isPk && !fk && (
                                      <span style={{ color: 'var(--qa-text-muted)', fontSize: '11px' }}>-</span>
                                    )}
                                  </div>
                                </td>
                                <td>
                                  {col.samples && col.samples.length > 0 ? (
                                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                                      {col.samples.slice(0, 3).map((s, idx) => (
                                        <span
                                          key={idx}
                                          style={{
                                            fontSize: '11px',
                                            fontFamily: 'var(--qa-font-mono)',
                                            color: 'var(--qa-text-secondary)',
                                            backgroundColor: 'var(--qa-bg-primary)',
                                            padding: '1px 6px',
                                            borderRadius: '3px',
                                            border: '1px solid var(--qa-border)',
                                          }}
                                        >
                                          {s}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <span style={{ color: 'var(--qa-text-muted)', fontSize: '11px' }}>No samples</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>

                      {/* Foreign Keys Notice */}
                      {relatedFks.length > 0 && (
                        <div
                          style={{
                            padding: '8px 12px',
                            borderRadius: 'var(--qa-radius-sm)',
                            backgroundColor: 'var(--qa-bg-elevated)',
                            border: '1px solid var(--qa-border)',
                            fontSize: '11.5px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                          }}
                        >
                          <Link2 size={13} color="var(--qa-accent)" />
                          <span style={{ fontWeight: 600 }}>Relationships:</span>
                          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                            {relatedFks.map((fk, idx) => (
                              <span key={idx} style={{ fontFamily: 'var(--qa-font-mono)', color: 'var(--qa-text-secondary)' }}>
                                {fk.from_table}.{fk.from_column} &rarr; {fk.to_table}.{fk.to_column}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* ER Diagram View */}
      {activeTab === 'diagram' && (
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1rem',
          }}
        >
          <SchemaDiagram metadata={metadata} databaseId={selectedDatabase} />
        </div>
      )}

      {/* Raw Prompt M-Schema View */}
      {activeTab === 'prompt' && (
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.25rem',
          }}
        >
          <div style={{ marginBottom: '10px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 700 }}>Prompt M-Schema Context</h3>
            <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
              This is the exact token format injected into the LLM system prompt for the "{selectedDatabase}" database.
            </p>
          </div>

          <pre
            style={{
              margin: 0,
              padding: '1rem',
              backgroundColor: 'var(--qa-bg-primary)',
              border: '1px solid var(--qa-border)',
              borderRadius: 'var(--qa-radius-sm)',
              fontSize: '12px',
              fontFamily: 'var(--qa-font-mono)',
              color: 'var(--qa-text-secondary)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.5,
              maxHeight: '600px',
              overflowY: 'auto',
            }}
          >
            {schemaPromptText || 'Loading schema prompt context...'}
          </pre>
        </div>
      )}
    </div>
  );
};
