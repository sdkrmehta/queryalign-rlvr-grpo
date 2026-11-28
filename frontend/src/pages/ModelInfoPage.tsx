import React from 'react';
import {
  Cpu,
  Shield,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { BENCHMARK_SUMMARY } from '../data/benchmarkData';

export const ModelInfoPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header Banner */}
      <div
        style={{
          backgroundColor: 'var(--qa-bg-card)',
          border: '1px solid var(--qa-border)',
          borderRadius: 'var(--qa-radius-md)',
          padding: '1.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              backgroundColor: 'var(--qa-accent-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--qa-accent)',
              border: '1px solid rgba(129, 150, 255, 0.3)',
            }}
          >
            <Cpu size={22} />
          </div>
          <div>
            <h1 style={{ fontSize: '20px', fontWeight: 800 }}>Model Specifications & RLVR Architecture</h1>
            <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)', marginTop: '2px' }}>
              Technical design, QLoRA adapter hyperparameters, GRPO reinforcement learning policy, and SQLite sandbox defense.
            </p>
          </div>
        </div>
      </div>

      {/* Grid: 4 Core Architectural Modules */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Module 1: Base Model & Fine-Tuning */}
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
            <Layers size={18} color="var(--qa-accent)" />
            <h2 style={{ fontSize: '15px', fontWeight: 700 }}>Base Foundation Model & QLoRA</h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Base Model</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>Qwen/Qwen2.5-Coder-1.5B-Instruct</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Parameters</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>1.54 Billion</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>QLoRA Rank (r)</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>16</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>QLoRA Alpha (&alpha;)</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>16 (Scaling = 1.0)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Target Modules</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>All Linear (q, k, v, o, gate, up, down)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Adapter SHA-256</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)', fontSize: '10px' }}>
                {BENCHMARK_SUMMARY.adapterSha256.substring(0, 24)}...
              </span>
            </div>
          </div>
        </div>

        {/* Module 2: GRPO Policy Optimization */}
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
            <Zap size={18} color="var(--qa-accent)" />
            <h2 style={{ fontSize: '15px', fontWeight: 700 }}>GRPO (Group Relative Policy Optimization)</h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Group Size (G)</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>8 Candidates per Query</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Exploration Sampling</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>T = 0.8, top_p = 0.95</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Advantage Formulation</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>Group Normalized: (R - &mu;) / &sigma;</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Critic Elimination</span>
              <span style={{ fontWeight: 600, color: 'var(--qa-success)' }}>100% Value Model-Free</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>KL Divergence Term</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>Token-level penalty vs SFT Ref</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Optimization Steps</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>100 Iterations</span>
            </div>
          </div>
        </div>

        {/* Module 3: Verifiable Reward Verification */}
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
            <Sparkles size={18} color="var(--qa-accent)" />
            <h2 style={{ fontSize: '15px', fontWeight: 700 }}>Verifiable Reward Formulation</h2>
          </div>

          <p style={{ fontSize: '12px', color: 'var(--qa-text-muted)' }}>
            Traditional RLHF uses black-box LLM judges that introduce hallucination loops. QueryAlign replaces judges with 100% mathematical verifiers:
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <div style={{ fontWeight: 700, color: 'var(--qa-accent)', fontFamily: 'var(--qa-font-mono)' }}>R_fmt &isin; [-0.50, +0.20]</div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>Strict regex validation of &lt;sql&gt;...&lt;/sql&gt; enclosure.</div>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <div style={{ fontWeight: 700, color: 'var(--qa-accent)', fontFamily: 'var(--qa-font-mono)' }}>R_ast &isin; [0.00, +0.20]</div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>SQLGlot AST parse validation. Strict read-only query policy.</div>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <div style={{ fontWeight: 700, color: 'var(--qa-accent)', fontFamily: 'var(--qa-font-mono)' }}>R_exec &isin; [-0.20, +0.30]</div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>Sandbox execution without runtime errors or deadline timeouts.</div>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <div style={{ fontWeight: 700, color: 'var(--qa-accent)', fontFamily: 'var(--qa-font-mono)' }}>R_result &isin; [0.00, +1.00]</div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>Multiset tuple hash equality against verified ground-truth.</div>
            </div>
            <div style={{ padding: '8px', backgroundColor: 'var(--qa-bg-elevated)', borderRadius: '4px', border: '1px solid var(--qa-border)' }}>
              <div style={{ fontWeight: 700, color: 'var(--qa-accent)', fontFamily: 'var(--qa-font-mono)' }}>R_eff &isin; [0.00, +0.30] (Package Composite Only)</div>
              <div style={{ fontSize: '11px', color: 'var(--qa-text-muted)' }}>Correctness-gated latency efficiency bonus in composite reward (inactive in NB3 4-reward run).</div>
            </div>
          </div>
        </div>

        {/* Module 4: SQLite Sandbox & Security Isolation */}
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
            <Shield size={18} color="var(--qa-success)" />
            <h2 style={{ fontSize: '15px', fontWeight: 700 }}>Deterministic Sandbox Security</h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>URI Mode</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>file:?mode=ro</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>PRAGMA Enforcement</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>PRAGMA query_only = ON</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>C-Authorizer Hook</span>
              <span style={{ fontWeight: 600, color: 'var(--qa-success)' }}>sqlite3_set_authorizer Active</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Denied Operations</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>INSERT, UPDATE, DELETE, ATTACH</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Watchdog Instruction Limit</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>2.0s Deadline via progress_handler</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--qa-border)' }}>
              <span style={{ color: 'var(--qa-text-muted)' }}>Max Bounded Rows</span>
              <span style={{ fontWeight: 600, fontFamily: 'var(--qa-font-mono)' }}>5,000 Rows Hard Limit</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
