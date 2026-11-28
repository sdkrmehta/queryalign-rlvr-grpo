// ============================================================================
// Genuine Benchmark Results from reports/benchmark_summary.json & reports/final_benchmark_report.md
// ============================================================================

export interface StageBenchmark {
  stageId: number;
  name: string;
  tag: string;
  model: string;
  executionAccuracy: number; // EX %
  sqlValidity: number; // %
  executionSuccess: number; // %
  description: string;
  keyMilestone: string;
}

export const PIPELINE_STAGES: StageBenchmark[] = [
  {
    stageId: 1,
    name: 'Zero-Shot Baseline',
    tag: 'Stage 1',
    model: 'Qwen2.5-Coder-1.5B-Instruct (Untuned)',
    executionAccuracy: 55.0,
    sqlValidity: 100.0,
    executionSuccess: 100.0,
    description: 'Greedy decoding from base model without domain tuning',
    keyMilestone: 'Baseline established on Spider 1.0 dev set',
  },
  {
    stageId: 2,
    name: 'QLoRA SFT Baseline',
    tag: 'Stage 2',
    model: 'QLoRA Rank 16, Alpha 16 (3 Epochs)',
    executionAccuracy: 58.0,
    sqlValidity: 100.0,
    executionSuccess: 100.0,
    description: 'Supervised fine-tuning on 7,000 Spider pairs for syntax & tag grounding',
    keyMilestone: 'Guaranteed <sql>...</sql> enclosure compliance',
  },
  {
    stageId: 3,
    name: 'RLVR Policy (Greedy Pass@1)',
    tag: 'Stage 3',
    model: 'GRPO Policy (100 Steps, G=8, 4 Rewards)',
    executionAccuracy: 66.0,
    sqlValidity: 96.0,
    executionSuccess: 78.0,
    description: 'Reinforcement learning with 4 verifiable rewards in SQLite sandbox',
    keyMilestone: '+11.0% absolute gain over Zero-Shot baseline',
  },
  {
    stageId: 4,
    name: 'Majority@8 Consensus Voting',
    tag: 'Stage 4',
    model: 'Test-Time Search (T=0.8, 8 Samples, Multiset Hash)',
    executionAccuracy: 80.0,
    sqlValidity: 96.0,
    executionSuccess: 96.0,
    description: 'Plurality voting over 8 stochastic candidates with multiset hash clustering',
    keyMilestone: '+14.0% test-time gain; 90.24% precision on consensus',
  },
];

export const BENCHMARK_SUMMARY = {
  benchmarkName: 'Spider 1.0 (Audited Dev Evaluation)',
  evalMode: 'first_50',
  nExamples: 50,
  adapterSha256: '9EF9C5326FC68CE4D9D602392F3DE107877C2C8083BC77A74D3B68799409ECF3',
  percentileMethod: 'linear (NumPy default, Hyndman & Fan Type 7)',
  greedy: {
    executionAccuracyPct: 66.0,
    sqlValidityPct: 78.0,
    execSuccessPct: 78.0,
    p50LatencyMs: 3566.965,
    p95LatencyMs: 7420.241,
  },
  majority8: {
    executionAccuracyPct: 80.0,
    deltaVsGreedyPts: 14.0,
    sqlValidityPct: 96.0,
    execSuccessPct: 96.0,
    p50LatencyMs: 7867.332,
    p95LatencyMs: 14295.827,
    selfCorrectionRatePct: 12.0,
    regressionRatePct: 6.0,
    decisionStatus: {
      CONSENSUS_ELECTED: {
        count: 41,
        pct: 82.0,
        accuracy: 90.24,
        meaning: 'Plurality consensus (K >= 3 identical multiset hashes). Safe for autonomous execution.',
      },
      LOW_CONFIDENCE: {
        count: 6,
        pct: 12.0,
        accuracy: 50.0,
        meaning: 'Winning cluster has weak support (K < 3). Route to human verification.',
      },
      NO_CONSENSUS_TIE: {
        count: 1,
        pct: 2.0,
        accuracy: 0.0,
        meaning: 'Tie between competing candidate clusters.',
      },
      ALL_FAILED: {
        count: 2,
        pct: 4.0,
        accuracy: 0.0,
        meaning: 'All 8 sampled candidates encountered AST rejection or runtime execution errors.',
      },
    },
  },
};

export const REWARD_FUNCTIONS_INFO = [
  {
    name: 'R_fmt (Format Validity)',
    bounds: '[-0.50, +0.20]',
    weight: 'Strict Enclosure',
    description: 'Enforces strict <sql>...</sql> tag structure without markdown noise.',
  },
  {
    name: 'R_ast (AST Security Policy)',
    bounds: '[0.00, +0.20]',
    weight: 'Static AST Verification',
    description: 'SQLGlot AST parse validation. Strictly enforces read-only SELECT-family statements.',
  },
  {
    name: 'R_exec (Execution Validity)',
    bounds: '[-0.20, +0.30]',
    weight: 'Sandbox Runtime',
    description: 'Ensures SQL executes cleanly in the deterministic SQLite sandbox within 2.0s deadline.',
  },
  {
    name: 'R_result (Multiset Accuracy)',
    bounds: '[0.00, +1.00]',
    weight: 'Ground Truth Verification',
    description: 'Order-independent multiset equivalence match against verified ground truth results.',
  },
  {
    name: 'R_eff (Latency Efficiency)',
    bounds: '[0.00, +0.10]',
    weight: 'Optional Bonus',
    description: 'Bonus reward for sandboxed execution completing in under 100 milliseconds.',
  },
];
