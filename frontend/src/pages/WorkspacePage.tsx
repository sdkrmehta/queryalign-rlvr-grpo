import React, { useState } from 'react';
import { Sparkles, Code2, Network, FileText, AlertCircle } from 'lucide-react';
import { DatabaseMetadata, InferenceMode, TextToSQLResponse } from '../types';
import { QueryComposer } from '../components/query/QueryComposer';
import { ResultsWorkspace } from '../components/results/ResultsWorkspace';
import { SchemaTree } from '../components/schema/SchemaTree';
import { SchemaDiagram } from '../components/schema/SchemaDiagram';
import { PIPELINE_STAGES } from '../data/benchmarkData';

interface WorkspacePageProps {
  question: string;
  setQuestion: (q: string) => void;
  selectedDatabase: string;
  setSelectedDatabase: (db: string) => void;
  databases: string[];
  mode: InferenceMode;
  setMode: (m: InferenceMode) => void;
  onGenerate: () => void;
  isLoading: boolean;
  onClear: () => void;
  schemaMetadata: DatabaseMetadata | null;
  schemaPromptText: string;
  lastResponse: TextToSQLResponse | null;
  editedSql: string;
  setEditedSql: (sql: string) => void;
  error: string | null;
}

export const WorkspacePage: React.FC<WorkspacePageProps> = ({
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
  schemaMetadata,
  schemaPromptText,
  lastResponse,
  editedSql,
  setEditedSql,
  error,
}) => {
  const [schemaView, setSchemaView] = useState<'tree' | 'diagram' | 'prompt'>('tree');

  const handleInsertTerm = (term: string) => {
    if (question.trim()) {
      setQuestion(`${question} ${term}`);
    } else {
      setQuestion(`List all ${term}`);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '1440px', margin: '0 auto' }}>
      {/* Alignment Workflow Pipeline Stage Breadcrumbs */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1rem 1.25rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={16} color="var(--qa-accent)" />
            <span style={{ fontSize: '13px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              End-to-End Alignment Workflow Pipeline
            </span>
          </div>
          <span style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
            Active Stage: {mode === 'greedy' ? 'Stage 3 (RLVR Greedy Policy)' : 'Stage 4 (Majority@8 Consensus)'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
          {PIPELINE_STAGES.map(stage => {
            const isCurrentlyActive =
              (mode === 'greedy' && stage.stageId === 3) ||
              (mode === 'majority8' && stage.stageId === 4);

            return (
              <div
                key={stage.stageId}
                style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--qa-radius-sm)',
                  backgroundColor: isCurrentlyActive ? 'var(--qa-accent-subtle)' : 'var(--qa-bg-elevated)',
                  border: isCurrentlyActive ? '1px solid var(--qa-accent)' : '1px solid var(--qa-border)',
                  position: 'relative',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span
                    style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      color: isCurrentlyActive ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                    }}
                  >
                    {stage.tag} {isCurrentlyActive && '• Active'}
                  </span>
                  <span
                    style={{
                      fontSize: '12px',
                      fontWeight: 800,
                      fontFamily: 'var(--qa-font-mono)',
                      color: stage.executionAccuracy >= 80 ? 'var(--qa-success)' : stage.executionAccuracy >= 65 ? 'var(--qa-accent)' : 'var(--qa-text-secondary)',
                    }}
                  >
                    {stage.executionAccuracy.toFixed(1)}% EX
                  </span>
                </div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--qa-text-primary)' }}>
                  {stage.name}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '2px', lineHeight: 1.3 }}>
                  {stage.keyMilestone}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Query Composer & Results on Left, Schema Explorer on Right */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.45fr) minmax(360px, 1fr)',
          gap: '1.5rem',
          alignItems: 'start',
        }}
      >
        {/* Left Column: Input Composer & Execution Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <QueryComposer
            question={question}
            setQuestion={setQuestion}
            selectedDatabase={selectedDatabase}
            setSelectedDatabase={setSelectedDatabase}
            databases={databases}
            mode={mode}
            setMode={setMode}
            onGenerate={onGenerate}
            isLoading={isLoading}
            onClear={onClear}
          />

          {/* Backend Error Alert if present */}
          {error && (
            <div
              style={{
                backgroundColor: 'var(--qa-danger-bg)',
                border: '1px solid var(--qa-danger-border)',
                borderRadius: 'var(--qa-radius-md)',
                padding: '1rem',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
                color: 'var(--qa-danger)',
              }}
            >
              <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <div style={{ fontWeight: 700, fontSize: '14px', marginBottom: '2px' }}>
                  Execution Pipeline Error
                </div>
                <div style={{ fontSize: '13px', lineHeight: 1.4 }}>{error}</div>
              </div>
            </div>
          )}

          {/* Results Workspace Component */}
          {lastResponse && (
            <ResultsWorkspace
              response={lastResponse}
              editedSql={editedSql}
              setEditedSql={setEditedSql}
            />
          )}
        </div>

        {/* Right Column: Schema Explorer with Tree, Diagram, and M-Schema Views */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            boxShadow: 'var(--qa-shadow-sm)',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Schema Explorer Header & View Switcher */}
          <div
            style={{
              padding: '0.85rem 1rem',
              borderBottom: '1px solid var(--qa-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              backgroundColor: 'var(--qa-bg-elevated)',
            }}
          >
            <div>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--qa-text-primary)' }}>
                Target Database Schema
              </div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', fontFamily: 'var(--qa-font-mono)' }}>
                {selectedDatabase}.sqlite ({schemaMetadata ? Object.keys(schemaMetadata.tables || {}).length : 0} tables)
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <button
                onClick={() => setSchemaView('tree')}
                className="qa-btn-ghost"
                title="Structured Columns & Types"
                style={{
                  padding: '4px 8px',
                  fontSize: '11px',
                  backgroundColor: schemaView === 'tree' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: schemaView === 'tree' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                }}
              >
                <Code2 size={13} />
                <span>Columns</span>
              </button>
              <button
                onClick={() => setSchemaView('diagram')}
                className="qa-btn-ghost"
                title="Entity-Relationship Diagram"
                style={{
                  padding: '4px 8px',
                  fontSize: '11px',
                  backgroundColor: schemaView === 'diagram' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: schemaView === 'diagram' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                }}
              >
                <Network size={13} />
                <span>ERD</span>
              </button>
              <button
                onClick={() => setSchemaView('prompt')}
                className="qa-btn-ghost"
                title="Prompt M-Schema Format"
                style={{
                  padding: '4px 8px',
                  fontSize: '11px',
                  backgroundColor: schemaView === 'prompt' ? 'var(--qa-accent-subtle)' : 'transparent',
                  color: schemaView === 'prompt' ? 'var(--qa-accent)' : 'var(--qa-text-muted)',
                }}
              >
                <FileText size={13} />
                <span>M-Schema</span>
              </button>
            </div>
          </div>

          {/* View Content */}
          <div style={{ maxHeight: '720px', overflowY: 'auto', padding: '0.75rem' }}>
            {schemaView === 'tree' && (
              <SchemaTree
                metadata={schemaMetadata}
                databaseId={selectedDatabase}
                onInsertTerm={handleInsertTerm}
              />
            )}

            {schemaView === 'diagram' && (
              <SchemaDiagram metadata={schemaMetadata} databaseId={selectedDatabase} />
            )}

            {schemaView === 'prompt' && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginBottom: '8px' }}>
                  Raw schema representation fed directly into model prompt context:
                </div>
                <pre
                  style={{
                    backgroundColor: 'var(--qa-bg-primary)',
                    border: '1px solid var(--qa-border)',
                    borderRadius: 'var(--qa-radius-sm)',
                    padding: '12px',
                    fontSize: '11.5px',
                    fontFamily: 'var(--qa-font-mono)',
                    color: 'var(--qa-text-secondary)',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    lineHeight: 1.5,
                  }}
                >
                  {schemaPromptText || 'Loading schema prompt...'}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
