import React from 'react';
import {
  BarChart3,
  ShieldCheck,
  Layers,
  Award,
  Vote,
  Hash,
} from 'lucide-react';
import {
  BENCHMARK_SUMMARY,
  PIPELINE_STAGES,
  REWARD_FUNCTIONS_INFO,
} from '../data/benchmarkData';

export const EvaluationPage: React.FC = () => {
  const summary = BENCHMARK_SUMMARY;

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header Banner */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
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
              <BarChart3 size={20} />
            </div>
            <div>
              <h1 style={{ fontSize: '20px', fontWeight: 800 }}>Spider 1.0 Benchmark & Evaluation Report</h1>
              <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
                Audited {summary.evalMode} evaluation slice ({summary.nExamples} examples across {summary.evalDatabasesNote}).
              </p>
            </div>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontFamily: 'var(--qa-font-mono)',
              fontSize: '11px',
              color: 'var(--qa-text-muted)',
              backgroundColor: 'var(--qa-bg-elevated)',
              padding: '6px 12px',
              borderRadius: 'var(--qa-radius-sm)',
              border: '1px solid var(--qa-border)',
            }}
          >
            <Hash size={13} />
            <span>Adapter SHA-256: {summary.adapterSha256.substring(0, 16)}...</span>
          </div>
        </div>
      </div>

      {/* Core Key Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
        {/* Card 1: Greedy Pass@1 */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--qa-text-muted)', fontSize: '12px' }}>
            <span>Greedy Pass@1 EX</span>
            <Layers size={16} />
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, marginTop: '8px', color: 'var(--qa-text-primary)' }}>
            {summary.greedy.executionAccuracyPct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-success)', marginTop: '4px', fontWeight: 600 }}>
            +{summary.matched100Slice.deltaPts.toFixed(1)} pp vs Zero-Shot on matched N={summary.matched100Slice.nExamples} slice ({summary.matched100Slice.grpoGreedyExPct.toFixed(1)}% vs {summary.matched100Slice.zeroShotExPct.toFixed(1)}%)
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
            P50 Latency: {summary.greedy.p50LatencyMs.toFixed(0)} ms
          </div>
        </div>

        {/* Card 2: Majority@8 Consensus */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-accent)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.25rem',
            boxShadow: '0 0 16px rgba(129, 150, 255, 0.12)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--qa-accent)', fontSize: '12px', fontWeight: 600 }}>
            <span>Majority@8 Consensus EX</span>
            <Vote size={16} />
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, marginTop: '8px', color: 'var(--qa-accent)' }}>
            {summary.majority8.executionAccuracyPct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-success)', marginTop: '4px', fontWeight: 600 }}>
            +{summary.majority8.deltaVsGreedyPts.toFixed(1)}% Test-Time Search Delta
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
            P50 Latency: {summary.majority8.p50LatencyMs.toFixed(0)} ms (8 Candidates)
          </div>
        </div>

        {/* Card 3: Consensus Precision */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--qa-text-muted)', fontSize: '12px' }}>
            <span>Consensus Precision</span>
            <Award size={16} />
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, marginTop: '8px', color: 'var(--qa-success)' }}>
            {summary.majority8.decisionStatus.CONSENSUS_ELECTED.accuracy.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-success)', marginTop: '4px', fontWeight: 600 }}>
            37 / 41 Correct on Agreement
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
            Safe for Autonomous Execution
          </div>
        </div>

        {/* Card 4: AST Security & Format */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--qa-text-muted)', fontSize: '12px' }}>
            <span>AST Security & Validity</span>
            <ShieldCheck size={16} color="var(--qa-success)" />
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, marginTop: '8px', color: 'var(--qa-text-primary)' }}>
            {summary.majority8.sqlValidityPct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-success)', marginTop: '4px', fontWeight: 600 }}>
            0 Sandbox DDL / DML Violations
          </div>
          <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
            Greedy Valid SQL: {summary.greedy.sqlValidityPct.toFixed(1)}% &rarr; Maj@8: {summary.majority8.sqlValidityPct.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Alignment Progression Matrix */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1.5rem',
        }}
      >
        <div style={{ marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 700 }}>Alignment Pipeline Progression Matrix</h2>
          <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
            Benchmarking performance across fine-tuning stages and search algorithms on Spider dev pairs.
          </p>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="qa-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Model Architecture & Config</th>
                <th style={{ textAlign: 'right' }}>Execution Accuracy (EX)</th>
                <th style={{ textAlign: 'right' }}>SQL Validity</th>
                <th style={{ textAlign: 'right' }}>Exec Success</th>
                <th>Key Engineering Milestone</th>
              </tr>
            </thead>
            <tbody>
              {PIPELINE_STAGES.map(stage => (
                <tr key={stage.stageId}>
                  <td>
                    <span
                      style={{
                        fontSize: '11px',
                        fontWeight: 700,
                        backgroundColor: 'var(--qa-bg-elevated)',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        border: '1px solid var(--qa-border)',
                      }}
                    >
                      {stage.tag}
                    </span>
                  </td>
                  <td>
                    <div style={{ fontWeight: 600, fontSize: '13px' }}>{stage.name}</div>
                    <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                      {stage.model} &bull; {stage.sampleScope}
                    </div>
                  </td>
                  <td style={{ textAlign: 'right', fontWeight: 800, fontFamily: 'var(--qa-font-mono)' }}>
                    <span
                      style={{
                        color:
                          stage.executionAccuracy === null
                            ? 'var(--qa-text-muted)'
                            : stage.executionAccuracy >= 80
                            ? 'var(--qa-success)'
                            : stage.executionAccuracy >= 65
                            ? 'var(--qa-accent)'
                            : 'var(--qa-text-primary)',
                      }}
                    >
                      {stage.executionAccuracy !== null ? `${stage.executionAccuracy.toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--qa-font-mono)', fontSize: '12px' }}>
                    {stage.sqlValidity !== null ? `${stage.sqlValidity.toFixed(1)}%` : 'N/A'}
                  </td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--qa-font-mono)', fontSize: '12px' }}>
                    {stage.executionSuccess !== null ? `${stage.executionSuccess.toFixed(1)}%` : 'N/A'}
                  </td>
                  <td style={{ fontSize: '12px', color: 'var(--qa-text-secondary)' }}>
                    {stage.keyMilestone}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Consensus Breakdown & Telemetry Analysis */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
        {/* Consensus Decision Status Distribution */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Vote size={18} color="var(--qa-accent)" />
            <h3 style={{ fontSize: '15px', fontWeight: 700 }}>Majority@8 Decision Telemetry</h3>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
            Categorization of test queries evaluated under 8-sample multiset hash clustering.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '6px' }}>
            {Object.entries(summary.majority8.decisionStatus).map(([key, item]) => {
              const isElected = key === 'CONSENSUS_ELECTED';
              const isLowConf = key === 'LOW_CONFIDENCE';

              return (
                <div
                  key={key}
                  style={{
                    padding: '10px 12px',
                    borderRadius: 'var(--qa-radius-sm)',
                    backgroundColor: 'var(--qa-bg-elevated)',
                    border: '1px solid var(--qa-border)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: isElected ? 'var(--qa-success)' : isLowConf ? 'var(--qa-warning)' : 'var(--qa-text-primary)' }}>
                      {key.replace(/_/g, ' ')}
                    </span>
                    <span style={{ fontSize: '12px', fontFamily: 'var(--qa-font-mono)', fontWeight: 700 }}>
                      {item.count} queries ({item.pct}%)
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', color: 'var(--qa-text-muted)' }}>
                    <span>{item.meaning}</span>
                    <span style={{ fontWeight: 700, color: item.accuracy >= 80 ? 'var(--qa-success)' : 'var(--qa-text-secondary)', marginLeft: '8px', flexShrink: 0 }}>
                      {item.accuracy.toFixed(1)}% Acc
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Verifiable Reward Formulations */}
        <div
          style={{
            backgroundColor: 'var(--qa-bg-card)',
            border: '1px solid var(--qa-border)',
            borderRadius: 'var(--qa-radius-md)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Award size={18} color="var(--qa-accent)" />
            <h3 style={{ fontSize: '15px', fontWeight: 700 }}>Verifiable Reward Functions</h3>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
            Zero LLM-as-a-judge: 100% deterministic mathematical rewards verified in sandbox.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
            {REWARD_FUNCTIONS_INFO.map(reward => (
              <div
                key={reward.name}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--qa-radius-sm)',
                  backgroundColor: 'var(--qa-bg-elevated)',
                  border: '1px solid var(--qa-border)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', fontWeight: 700, fontFamily: 'var(--qa-font-mono)', color: 'var(--qa-accent)' }}>
                    {reward.name}
                  </span>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--qa-font-mono)', color: 'var(--qa-text-muted)' }}>
                    {reward.bounds}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--qa-text-secondary)', marginTop: '2px' }}>
                  {reward.description}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
