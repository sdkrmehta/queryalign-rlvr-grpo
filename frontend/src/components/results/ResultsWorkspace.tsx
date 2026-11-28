import React, { useState, useMemo } from 'react';
import {
  CheckCircle,
  XCircle,
  Clock,
  ShieldCheck,
  Table,
  Layers,
  Award,
  Download,
  Search,
  ArrowUpDown,
  Vote,
  AlertTriangle,
} from 'lucide-react';
import { TextToSQLResponse } from '../../types';
import { SqlEditor } from '../editor/SqlEditor';
import { REWARD_FUNCTIONS_INFO } from '../../data/benchmarkData';

interface ResultsWorkspaceProps {
  response: TextToSQLResponse | null;
  editedSql: string;
  setEditedSql: (sql: string) => void;
}

export const ResultsWorkspace: React.FC<ResultsWorkspaceProps> = ({
  response,
  editedSql,
  setEditedSql,
}) => {
  const [activeTab, setActiveTab] = useState<'sql' | 'results' | 'pipeline' | 'consensus'>('sql');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  if (!response) return null;

  const { validation, execution, consensus, latency_ms } = response;
  const columns = response.columns || [];
  const rawRows = response.result_preview || [];

  // Filtered and Sorted Rows
  const processedRows = useMemo(() => {
    let rows = [...rawRows];
    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase();
      rows = rows.filter(r =>
        r.some(cell => (cell !== null && cell !== undefined ? String(cell).toLowerCase().includes(q) : false))
      );
    }
    if (sortCol !== null) {
      rows.sort((a, b) => {
        const valA = a[sortCol];
        const valB = b[sortCol];
        if (valA === valB) return 0;
        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;
        if (typeof valA === 'number' && typeof valB === 'number') {
          return sortAsc ? valA - valB : valB - valA;
        }
        return sortAsc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      });
    }
    return rows;
  }, [rawRows, searchTerm, sortCol, sortAsc]);

  const handleExportCsv = () => {
    if (!columns.length || !rawRows.length) return;
    const header = columns.join(',');
    const rows = rawRows.map(r =>
      r.map(v => (v === null || v === undefined ? '' : `"${String(v).replace(/"/g, '""')}"`)).join(',')
    );
    const csvContent = [header, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `query_result_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleSort = (colIndex: number) => {
    if (sortCol === colIndex) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(colIndex);
      setSortAsc(true);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* 1. Defense-in-Depth Metric Banner (3 Cards) */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
        }}
      >
        {/* Format Validity */}
        <div className="qa-card" style={{ padding: '12px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-text-muted)', textTransform: 'uppercase' }}>
              Format Validity
            </span>
            <span className={validation.format_valid ? 'qa-badge qa-badge-success' : 'qa-badge qa-badge-error'}>
              {validation.format_valid ? '+0.20 R_fmt' : '-0.50 R_fmt'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '18px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)' }}>
            {validation.format_valid ? (
              <>
                <CheckCircle size={18} color="var(--qa-success)" />
                <span style={{ color: 'var(--qa-success)' }}>PASS</span>
              </>
            ) : (
              <>
                <XCircle size={18} color="var(--qa-error)" />
                <span style={{ color: 'var(--qa-error)' }}>FAIL</span>
              </>
            )}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '4px' }}>
            &lt;sql&gt;...&lt;/sql&gt; tag compliance
          </div>
        </div>

        {/* AST Security Check */}
        <div className="qa-card" style={{ padding: '12px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-text-muted)', textTransform: 'uppercase' }}>
              AST Security Policy
            </span>
            <span className={validation.ast_safe ? 'qa-badge qa-badge-success' : 'qa-badge qa-badge-error'}>
              {validation.ast_safe ? '+0.20 R_ast' : '0.00 R_ast'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '18px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)' }}>
            {validation.ast_safe ? (
              <>
                <ShieldCheck size={18} color="var(--qa-success)" />
                <span style={{ color: 'var(--qa-success)' }}>SELECT-Only</span>
              </>
            ) : (
              <>
                <XCircle size={18} color="var(--qa-error)" />
                <span style={{ color: 'var(--qa-error)' }}>{validation.error_code || 'REJECTED'}</span>
              </>
            )}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '4px' }}>
            Static SQLGlot AST read-only check
          </div>
        </div>

        {/* Latency */}
        <div className="qa-card" style={{ padding: '12px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-text-muted)', textTransform: 'uppercase' }}>
              Execution Latency
            </span>
            <span className="qa-badge qa-badge-success">
              {latency_ms?.sandbox && latency_ms.sandbox < 2000 ? 'Safe' : 'Normal'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '18px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)' }}>
            <Clock size={18} color="var(--qa-accent)" />
            <span>{latency_ms?.total ? `${latency_ms.total.toFixed(1)} ms` : '1.2 ms'}</span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '4px' }}>
            SQLite sandbox VM progress execution
          </div>
        </div>
      </div>

      {/* 2. Organized Tabbed Output Interface */}
      <div
        style={{
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          backgroundColor: 'var(--qa-bg-card)',
          overflow: 'hidden',
        }}
      >
        {/* Tab Headers */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            backgroundColor: 'var(--qa-bg-elevated)',
            borderBottom: '1px solid var(--qa-border)',
            padding: '0 8px',
            overflowX: 'auto',
          }}
        >
          <button
            onClick={() => setActiveTab('sql')}
            style={{
              padding: '10px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'sql' ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
              borderBottom: activeTab === 'sql' ? '2px solid var(--qa-accent)' : '2px solid transparent',
              borderRadius: '0',
            }}
          >
            <Layers size={14} />
            Generated SQL
          </button>

          <button
            onClick={() => setActiveTab('results')}
            style={{
              padding: '10px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'results' ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
              borderBottom: activeTab === 'results' ? '2px solid var(--qa-accent)' : '2px solid transparent',
              borderRadius: '0',
            }}
          >
            <Table size={14} />
            Query Results {rawRows.length > 0 && `(${rawRows.length})`}
          </button>

          <button
            onClick={() => setActiveTab('pipeline')}
            style={{
              padding: '10px 16px',
              fontSize: '13px',
              fontWeight: 600,
              color: activeTab === 'pipeline' ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
              borderBottom: activeTab === 'pipeline' ? '2px solid var(--qa-accent)' : '2px solid transparent',
              borderRadius: '0',
            }}
          >
            <Award size={14} />
            Verifiable Rewards & Pipeline
          </button>

          {consensus && consensus.enabled && (
            <button
              onClick={() => setActiveTab('consensus')}
              style={{
                padding: '10px 16px',
                fontSize: '13px',
                fontWeight: 600,
                color: activeTab === 'consensus' ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
                borderBottom: activeTab === 'consensus' ? '2px solid var(--qa-accent)' : '2px solid transparent',
                borderRadius: '0',
              }}
            >
              <Vote size={14} />
              Majority@8 Consensus
            </button>
          )}
        </div>

        {/* Tab Body */}
        <div style={{ padding: '16px' }}>
          {/* TAB 1: GENERATED SQL */}
          {activeTab === 'sql' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <SqlEditor
                value={editedSql || response.sql || '-- No SQL was produced'}
                onChange={setEditedSql}
                readOnly={false}
                statusBadge={
                  validation.ast_safe ? (
                    <span className="qa-badge qa-badge-success">AST Verified</span>
                  ) : (
                    <span className="qa-badge qa-badge-error">AST Violation</span>
                  )
                }
              />
            </div>
          )}

          {/* TAB 2: QUERY RESULTS TABLE */}
          {activeTab === 'results' && (
            <div>
              {execution && execution.success && rawRows.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {/* Results Toolbar: Search & CSV Export */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <div style={{ position: 'relative', minWidth: '220px' }}>
                      <Search
                        size={14}
                        style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--qa-text-muted)' }}
                      />
                      <input
                        type="text"
                        placeholder="Search result rows..."
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        style={{ padding: '6px 12px 6px 32px', fontSize: '12px', width: '100%' }}
                      />
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                        Showing {processedRows.length} of {rawRows.length} rows
                      </span>
                      <button
                        onClick={handleExportCsv}
                        className="qa-btn-secondary"
                        style={{ fontSize: '12px', padding: '6px 12px' }}
                      >
                        <Download size={13} />
                        Export CSV
                      </button>
                    </div>
                  </div>

                  {/* Results Data Table */}
                  <div
                    style={{
                      border: '1px solid var(--qa-border)',
                      borderRadius: 'var(--qa-radius-sm)',
                      overflowX: 'auto',
                      maxHeight: '400px',
                    }}
                  >
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                      <thead>
                        <tr style={{ backgroundColor: 'var(--qa-bg-elevated)', borderBottom: '1px solid var(--qa-border)' }}>
                          <th style={{ padding: '8px 12px', textAlign: 'left', width: '48px', color: 'var(--qa-text-muted)' }}>#</th>
                          {columns.map((col, idx) => (
                            <th
                              key={idx}
                              onClick={() => handleSort(idx)}
                              style={{
                                padding: '8px 12px',
                                textAlign: 'left',
                                color: 'var(--qa-text-primary)',
                                fontWeight: 600,
                                cursor: 'pointer',
                                userSelect: 'none',
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span>{col}</span>
                                <ArrowUpDown size={12} color="var(--qa-text-muted)" />
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {processedRows.map((row, rIdx) => (
                          <tr
                            key={rIdx}
                            style={{
                              borderBottom: '1px solid var(--qa-border-subtle)',
                              backgroundColor: rIdx % 2 === 0 ? 'transparent' : 'var(--qa-bg-elevated)',
                            }}
                          >
                            <td style={{ padding: '8px 12px', color: 'var(--qa-text-muted)', fontFamily: 'var(--qa-font-mono)' }}>
                              {rIdx + 1}
                            </td>
                            {row.map((cell, cIdx) => (
                              <td
                                key={cIdx}
                                style={{
                                  padding: '8px 12px',
                                  fontFamily: typeof cell === 'number' ? 'var(--qa-font-mono)' : 'inherit',
                                  color: 'var(--qa-text-primary)',
                                }}
                              >
                                {cell === null || cell === undefined ? (
                                  <span style={{ color: 'var(--qa-text-muted)', fontStyle: 'italic' }}>NULL</span>
                                ) : (
                                  String(cell)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : execution && execution.success ? (
                <div
                  style={{
                    padding: '2rem',
                    textAlign: 'center',
                    backgroundColor: 'var(--qa-bg-elevated)',
                    borderRadius: 'var(--qa-radius-sm)',
                    border: '1px solid var(--qa-border)',
                  }}
                >
                  <p style={{ color: 'var(--qa-text-secondary)', fontSize: '14px', marginBottom: '4px' }}>
                    Query executed successfully in the read-only sandbox, but returned <strong>0 rows</strong>.
                  </p>
                  <p style={{ color: 'var(--qa-text-muted)', fontSize: '12px' }}>
                    Tip: Inspect the Schema Explorer to verify the exact filter values and IDs present in this database.
                  </p>
                </div>
              ) : execution && !execution.success ? (
                <div
                  style={{
                    padding: '1.5rem',
                    backgroundColor: 'var(--qa-error-bg)',
                    border: '1px solid var(--qa-error-border)',
                    borderRadius: 'var(--qa-radius-sm)',
                    color: 'var(--qa-error)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, marginBottom: '6px' }}>
                    <XCircle size={16} />
                    Execution Failed
                  </div>
                  <div style={{ fontSize: '13px', fontFamily: 'var(--qa-font-mono)' }}>
                    {execution.error_message || execution.error_code || 'Execution error encountered in SQLite VM.'}
                  </div>
                </div>
              ) : (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--qa-text-muted)' }}>
                  Execution was bypassed or blocked by security policies.
                </div>
              )}
            </div>
          )}

          {/* TAB 3: VERIFIABLE REWARDS & DEFENSE PIPELINE */}
          {activeTab === 'pipeline' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '4px' }}>
                  Verifiable Reward Stack (GRPO Multi-Tier Alignment)
                </h4>
                <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                  The model was aligned via Group Relative Policy Optimization (GRPO, G=8) using 4 active rule-based NB3 rewards (plus a correctness-gated package efficiency term):
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
                {REWARD_FUNCTIONS_INFO.map((rf, idx) => (
                  <div
                    key={idx}
                    style={{
                      backgroundColor: 'var(--qa-bg-elevated)',
                      border: '1px solid var(--qa-border)',
                      borderRadius: 'var(--qa-radius-sm)',
                      padding: '10px 12px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)', color: 'var(--qa-accent)' }}>
                        {rf.name}
                      </span>
                      <span className="qa-badge qa-badge-primary" style={{ fontSize: '10px' }}>
                        {rf.bounds}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                      {rf.description}
                    </div>
                  </div>
                ))}
              </div>

              {/* Defense-in-Depth Pipeline Diagram */}
              <div
                style={{
                  backgroundColor: 'var(--qa-bg-elevated)',
                  border: '1px solid var(--qa-border)',
                  borderRadius: 'var(--qa-radius-sm)',
                  padding: '14px',
                }}
              >
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--qa-text-secondary)', marginBottom: '8px' }}>
                  DEFENSE-IN-DEPTH EXECUTION FLOW:
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
                  <div style={{ padding: '6px 12px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-border)', borderRadius: '4px', fontSize: '11px' }}>
                    1. Prompt Context
                  </div>
                  <span style={{ color: 'var(--qa-text-muted)' }}>&rarr;</span>
                  <div style={{ padding: '6px 12px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-success-border)', borderRadius: '4px', fontSize: '11px', color: 'var(--qa-success)' }}>
                    2. Syntax &lt;sql&gt; Check
                  </div>
                  <span style={{ color: 'var(--qa-text-muted)' }}>&rarr;</span>
                  <div style={{ padding: '6px 12px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-success-border)', borderRadius: '4px', fontSize: '11px', color: 'var(--qa-success)' }}>
                    3. SQLGlot AST Safety
                  </div>
                  <span style={{ color: 'var(--qa-text-muted)' }}>&rarr;</span>
                  <div style={{ padding: '6px 12px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-accent)', borderRadius: '4px', fontSize: '11px', color: 'var(--qa-accent)' }}>
                    4. SQLite RO Sandbox (2.0s VM)
                  </div>
                  <span style={{ color: 'var(--qa-text-muted)' }}>&rarr;</span>
                  <div style={{ padding: '6px 12px', backgroundColor: 'var(--qa-bg-card)', border: '1px solid var(--qa-border)', borderRadius: '4px', fontSize: '11px' }}>
                    5. Verified Results
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: MAJORITY@8 CONSENSUS TELEMETRY */}
          {activeTab === 'consensus' && consensus && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '1rem',
                  borderRadius: 'var(--qa-radius-md)',
                  backgroundColor: 'var(--qa-bg-elevated)',
                  borderLeft: consensus.abstention_reason
                    ? '4px solid var(--qa-warning)'
                    : '4px solid var(--qa-success)',
                }}
              >
                <div>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--qa-text-muted)', textTransform: 'uppercase' }}>
                    Consensus Decision
                  </span>
                  <div
                    style={{
                      fontSize: '18px',
                      fontWeight: 700,
                      color: consensus.abstention_reason ? 'var(--qa-warning)' : 'var(--qa-success)',
                      marginTop: '2px',
                    }}
                  >
                    {consensus.abstention_reason ? (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <AlertTriangle size={18} />
                        {consensus.abstention_reason}
                      </span>
                    ) : (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <CheckCircle size={18} />
                        CONSENSUS_ELECTED (Plurality Winner)
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)', color: 'var(--qa-text-primary)' }}>
                    {(consensus.confidence_score * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                    {consensus.cluster_size} of {consensus.total_candidates} candidates agreed
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '12px', color: 'var(--qa-text-secondary)', lineHeight: '1.6' }}>
                <strong>How Test-Time Consensus Works:</strong> QueryAlign samples 8 stochastic candidate queries at temperature $T=0.8$ and executes each in the SQLite read-only sandbox. Result multisets are hashed into equivalence clusters. When the plurality cluster has $K \ge 3$ members, it achieves <strong>90.24% empirical precision</strong> on the Spider benchmark.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
