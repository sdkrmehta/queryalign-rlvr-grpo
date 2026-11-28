import React, { useState } from 'react';
import { DatabaseMetadata } from '../../types';
import { Key, Link2, Table as TableIcon } from 'lucide-react';

interface SchemaDiagramProps {
  metadata: DatabaseMetadata | null;
  databaseId: string;
}

export const SchemaDiagram: React.FC<SchemaDiagramProps> = ({
  metadata,
  databaseId,
}) => {
  const [direction, setDirection] = useState<'row' | 'column'>('row');

  if (!metadata || !metadata.tables) {
    return (
      <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--qa-text-muted)' }}>
        No relational metadata available for {databaseId}.
      </div>
    );
  }

  const tables = Object.entries(metadata.tables);
  const foreignKeys = metadata.foreign_keys || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Controls Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--qa-text-secondary)' }}>
          {tables.length} Entities &nbsp;|&nbsp; {foreignKeys.length} Foreign Key Relations
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>Layout:</span>
          <select
            value={direction}
            onChange={e => setDirection(e.target.value as 'row' | 'column')}
            style={{ padding: '3px 8px', fontSize: '11px' }}
          >
            <option value="row">Grid / Horizontal</option>
            <option value="column">Stacked / Vertical</option>
          </select>
        </div>
      </div>

      {/* Relational Foreign Key Flow Chips */}
      {foreignKeys.length > 0 && (
        <div style={{ backgroundColor: 'var(--qa-bg-elevated)', padding: '10px 12px', borderRadius: 'var(--qa-radius-sm)', border: '1px solid var(--qa-border)' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
            Foreign Key Flow Paths:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {foreignKeys.map((fk, idx) => (
              <span
                key={idx}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: 'var(--qa-bg-card)',
                  border: '1px solid var(--qa-border)',
                  borderRadius: 'var(--qa-radius-sm)',
                  padding: '4px 10px',
                  fontSize: '11px',
                  fontFamily: 'var(--qa-font-mono)',
                  color: 'var(--qa-text-secondary)',
                }}
              >
                <Link2 size={12} color="var(--qa-accent)" />
                <strong style={{ color: 'var(--qa-accent)' }}>
                  {fk.from_table}.{fk.from_column}
                </strong>
                &rarr;
                <span>
                  {fk.to_table}.{fk.to_column}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Table Entity Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: direction === 'row' ? 'repeat(auto-fit, minmax(240px, 1fr))' : '1fr',
          gap: '14px',
        }}
      >
        {tables.map(([tName, tData]) => {
          const pks = new Set(tData.primary_keys.map(k => String(k).toLowerCase()));

          return (
            <div
              key={tName}
              style={{
                backgroundColor: 'var(--qa-bg-card)',
                border: '1px solid var(--qa-border)',
                borderRadius: 'var(--qa-radius-md)',
                overflow: 'hidden',
                boxShadow: 'var(--qa-shadow-sm)',
              }}
            >
              {/* Entity Title */}
              <div
                style={{
                  backgroundColor: 'var(--qa-bg-elevated)',
                  borderBottom: '1px solid var(--qa-border)',
                  padding: '8px 12px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '13px' }}>
                  <TableIcon size={14} color="var(--qa-accent)" />
                  <span>{tName}</span>
                </div>
                <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                  {tData.columns.length} attributes
                </span>
              </div>

              {/* Attributes */}
              <div style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {tData.columns.map(c => {
                  const isPk = pks.has(c.name.toLowerCase()) || c.is_pk;
                  const isFk = foreignKeys.some(
                    fk =>
                      fk.from_table.toLowerCase() === tName.toLowerCase() &&
                      fk.from_column.toLowerCase() === c.name.toLowerCase()
                  );

                  return (
                    <div
                      key={c.name}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: '12px',
                        fontFamily: 'var(--qa-font-mono)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {isPk ? (
                          <Key size={12} color="var(--qa-warning)" />
                        ) : isFk ? (
                          <Link2 size={12} color="var(--qa-accent)" />
                        ) : (
                          <span style={{ width: '12px' }} />
                        )}
                        <span style={{ fontWeight: isPk ? 700 : 500, color: isPk ? 'var(--qa-warning)' : 'var(--qa-text-primary)' }}>
                          {c.name}
                        </span>
                      </div>
                      <span style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                        {c.type}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
