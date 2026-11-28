import React, { useState } from 'react';
import { Database, ChevronRight, ChevronDown, Key, Link2, Search, Table } from 'lucide-react';
import { DatabaseMetadata } from '../../types';

interface SchemaTreeProps {
  metadata: DatabaseMetadata | null;
  databaseId: string;
  onInsertTerm?: (term: string) => void;
}

export const SchemaTree: React.FC<SchemaTreeProps> = ({
  metadata,
  databaseId,
  onInsertTerm,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedTables, setExpandedTables] = useState<Record<string, boolean>>({});

  if (!metadata || !metadata.tables) {
    return (
      <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--qa-text-muted)', fontSize: '13px' }}>
        No schema metadata available for "{databaseId}".
      </div>
    );
  }

  const tableEntries = Object.entries(metadata.tables);
  const foreignKeys = metadata.foreign_keys || [];

  const toggleTable = (tName: string) => {
    setExpandedTables(prev => ({ ...prev, [tName]: !prev[tName] }));
  };

  const filteredTables = tableEntries.filter(([tName, tData]) => {
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    if (tName.toLowerCase().includes(term)) return true;
    return tData.columns.some(c => c.name.toLowerCase().includes(term));
  });

  return (
    <div
      style={{
        border: '1px solid var(--qa-border)',
        borderRadius: 'var(--qa-radius-md)',
        backgroundColor: 'var(--qa-bg-card)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Search Header */}
      <div
        style={{
          padding: '10px 14px',
          backgroundColor: 'var(--qa-bg-elevated)',
          borderBottom: '1px solid var(--qa-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '10px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={15} color="var(--qa-accent)" />
          <span style={{ fontSize: '13px', fontWeight: 600 }}>{databaseId}</span>
          <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
            ({tableEntries.length} tables, {foreignKeys.length} relations)
          </span>
        </div>

        <div style={{ position: 'relative', width: '160px' }}>
          <Search size={13} style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: 'var(--qa-text-muted)' }} />
          <input
            type="text"
            placeholder="Filter tables..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            style={{ width: '100%', padding: '4px 8px 4px 26px', fontSize: '11px' }}
          />
        </div>
      </div>

      {/* Tables List */}
      <div style={{ maxHeight: '420px', overflowY: 'auto', padding: '8px' }}>
        {filteredTables.map(([tName, tData]) => {
          const isExpanded = expandedTables[tName] ?? true; // Default expanded
          const pks = new Set(tData.primary_keys.map(k => String(k).toLowerCase()));

          // Foreign keys originating from this table
          const tableFks = foreignKeys.filter(
            fk => fk.from_table.toLowerCase() === tName.toLowerCase()
          );

          return (
            <div
              key={tName}
              style={{
                marginBottom: '8px',
                border: '1px solid var(--qa-border-subtle)',
                borderRadius: 'var(--qa-radius-sm)',
                backgroundColor: 'var(--qa-bg-subtle)',
                overflow: 'hidden',
              }}
            >
              {/* Table Header Row */}
              <div
                onClick={() => toggleTable(tName)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 10px',
                  cursor: 'pointer',
                  backgroundColor: 'var(--qa-bg-elevated)',
                  userSelect: 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {isExpanded ? <ChevronDown size={14} color="var(--qa-text-muted)" /> : <ChevronRight size={14} color="var(--qa-text-muted)" />}
                  <Table size={14} color="var(--qa-accent)" />
                  <span style={{ fontWeight: 600, fontSize: '13px', color: 'var(--qa-text-primary)' }}>
                    {tName}
                  </span>
                </div>
                <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                  {tData.columns.length} cols
                </span>
              </div>

              {/* Table Columns (if expanded) */}
              {isExpanded && (
                <div style={{ padding: '6px 10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {tData.columns.map(c => {
                    const isPk = pks.has(c.name.toLowerCase()) || c.is_pk;
                    const fkObj = tableFks.find(
                      fk => fk.from_column.toLowerCase() === c.name.toLowerCase()
                    );

                    return (
                      <div
                        key={c.name}
                        onClick={() => onInsertTerm?.(c.name)}
                        title={`Click to insert "${c.name}" into question`}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '4px 6px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          cursor: onInsertTerm ? 'pointer' : 'default',
                          transition: 'background-color 0.1s ease',
                        }}
                        onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'var(--qa-bg-elevated)')}
                        onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ fontFamily: 'var(--qa-font-mono)', fontWeight: 500, color: 'var(--qa-text-primary)' }}>
                            {c.name}
                          </span>
                          <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)', fontFamily: 'var(--qa-font-mono)' }}>
                            {c.type}
                          </span>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          {isPk && (
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '2px',
                                fontSize: '10px',
                                fontWeight: 700,
                                padding: '1px 5px',
                                borderRadius: '3px',
                                backgroundColor: 'var(--qa-warning-bg)',
                                color: 'var(--qa-warning)',
                                border: '1px solid var(--qa-warning-border)',
                              }}
                            >
                              <Key size={10} /> PK
                            </span>
                          )}

                          {fkObj && (
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '2px',
                                fontSize: '10px',
                                fontWeight: 600,
                                padding: '1px 5px',
                                borderRadius: '3px',
                                backgroundColor: 'var(--qa-accent-subtle)',
                                color: 'var(--qa-accent)',
                                border: '1px solid rgba(129, 150, 255, 0.3)',
                              }}
                            >
                              <Link2 size={10} /> FK &rarr; {fkObj.to_table}.{fkObj.to_column}
                            </span>
                          )}

                          {c.samples && c.samples.length > 0 && (
                            <span style={{ fontSize: '10px', color: 'var(--qa-text-muted)', marginLeft: '4px' }}>
                              [{c.samples.slice(0, 2).map(s => `'${s}'`).join(', ')}]
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
