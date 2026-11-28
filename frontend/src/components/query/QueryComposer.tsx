import React from 'react';
import { Sparkles, Trash2, Cpu, Database, RefreshCw } from 'lucide-react';
import { InferenceMode } from '../../types';
import { SAMPLE_QUESTIONS } from '../../data/sampleQuestions';

interface QueryComposerProps {
  question: string;
  setQuestion: (val: string) => void;
  selectedDatabase: string;
  setSelectedDatabase: (db: string) => void;
  databases: string[];
  mode: InferenceMode;
  setMode: (mode: InferenceMode) => void;
  onGenerate: () => void;
  isLoading: boolean;
  onClear: () => void;
  onDatabaseChange?: (db: string) => void;
}

export const QueryComposer: React.FC<QueryComposerProps> = ({
  question,
  setQuestion,
  selectedDatabase,
  setSelectedDatabase,
  databases,
  mode,
  setMode,
  onGenerate,
  isLoading,
  onClear,
  onDatabaseChange,
}) => {
  const sampleList = SAMPLE_QUESTIONS[selectedDatabase] || [];

  return (
    <div
      style={{
        backgroundColor: 'var(--qa-bg-card)',
        border: '1px solid var(--qa-border)',
        borderRadius: 'var(--qa-radius-md)',
        padding: '1.25rem',
        boxShadow: 'var(--qa-shadow-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
      }}
    >
      {/* Controls Bar: Database & Mode Selection */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          paddingBottom: '12px',
          borderBottom: '1px solid var(--qa-border-subtle)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={15} color="var(--qa-accent)" />
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)' }}>
            Target Database:
          </span>
          <select
            value={selectedDatabase}
            onChange={e => {
              const val = e.target.value;
              setSelectedDatabase(val);
              onDatabaseChange?.(val);
            }}
            style={{
              padding: '5px 10px',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            {databases.map(db => (
              <option key={db} value={db}>
                {db}
              </option>
            ))}
          </select>
        </div>

        {/* Inference Mode Toggle: Greedy vs Majority@8 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={15} color="var(--qa-accent)" />
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--qa-text-primary)', marginRight: '4px' }}>
            Mode:
          </span>

          <div
            style={{
              display: 'inline-flex',
              backgroundColor: 'var(--qa-bg-elevated)',
              border: '1px solid var(--qa-border)',
              borderRadius: 'var(--qa-radius-sm)',
              padding: '2px',
            }}
          >
            <button
              onClick={() => setMode('greedy')}
              style={{
                padding: '4px 10px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '4px',
                backgroundColor: mode === 'greedy' ? 'var(--qa-accent)' : 'transparent',
                color: mode === 'greedy' ? '#FFFFFF' : 'var(--qa-text-secondary)',
              }}
              title="Deterministic Pass@1 Greedy Decoding (66.0% EX)"
            >
              Greedy Pass@1
            </button>
            <button
              onClick={() => setMode('majority8')}
              style={{
                padding: '4px 10px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '4px',
                backgroundColor: mode === 'majority8' ? 'var(--qa-accent)' : 'transparent',
                color: mode === 'majority8' ? '#FFFFFF' : 'var(--qa-text-secondary)',
              }}
              title="Majority@8 Plurality Voting Consensus (80.0% EX, +14.0% test-time compute)"
            >
              Majority@8 (Consensus)
            </button>
          </div>
        </div>
      </div>

      {/* Suggested Quick Question Chips */}
      {sampleList.length > 0 && (
        <div>
          <div
            style={{
              fontSize: '11px',
              fontWeight: 700,
              color: 'var(--qa-text-muted)',
              marginBottom: '6px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Suggested Benchmark Queries:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {sampleList.slice(0, 4).map((q, idx) => (
              <button
                key={idx}
                onClick={() => setQuestion(q)}
                className="qa-btn-ghost"
                style={{
                  fontSize: '12px',
                  padding: '4px 10px',
                  backgroundColor: 'var(--qa-bg-elevated)',
                  border: '1px solid var(--qa-border)',
                  borderRadius: 'var(--qa-radius-sm)',
                  color: 'var(--qa-text-secondary)',
                  textAlign: 'left',
                }}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Natural Language Query Textarea */}
      <div style={{ position: 'relative' }}>
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder={`e.g. List all employee names with a salary greater than 80000 in ${selectedDatabase}...`}
          rows={3}
          style={{
            width: '100%',
            padding: '12px',
            fontSize: '14px',
            lineHeight: '1.5',
            resize: 'vertical',
          }}
        />
      </div>

      {/* Bottom Actions Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button
          onClick={onClear}
          className="qa-btn-ghost"
          disabled={!question && !isLoading}
          style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}
        >
          <Trash2 size={14} />
          Clear Input
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={onGenerate}
            disabled={!question.trim() || isLoading}
            className="qa-btn-primary"
            style={{ padding: '8px 20px', fontSize: '13px' }}
          >
            {isLoading ? (
              <>
                <RefreshCw size={15} className="qa-spin" style={{ animation: 'spin 1s linear infinite' }} />
                Generating & Executing...
              </>
            ) : (
              <>
                <Sparkles size={15} />
                Generate & Execute SQL
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
