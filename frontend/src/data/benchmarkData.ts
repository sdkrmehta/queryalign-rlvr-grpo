// ============================================================================
// Genuine Benchmark Results from reports/benchmark_summary.json & artifact JSONs
// ============================================================================

export interface StageBenchmark {
  stageId: number;
  name: string;
  tag: string;
  model: string;
  sampleScope: string;
  executionAccuracy: number | null; // EX % (null when no saved eval artifact exists)
  sqlValidity: number | null; // %
  executionSuccess: number | null; // %
  description: string;
  keyMilestone: string;
}

export const PIPELINE_STAGES: StageBenchmark[] = [
  {
    stageId: 1,
    name: 'Zero-Shot Baseline',
    tag: 'Stage 1',
    model: 'Qwen2.5-Coder-1.5B-Instruct (Untuned)',
    sampleScope: 'Spider dev[:100] (N=100)',
    executionAccuracy: 55.0,
    sqlValidity: 100.0,
    executionSuccess: null,
    description: 'Greedy decoding from base model on Spider dev[:100] (N=100); valid_sql_pct reflects tag extraction',
    keyMilestone: 'Untuned baseline recorded on Spider dev[:100] (55/100 EX)',
  },
  {
    stageId: 2,
    name: 'QLoRA SFT Baseline',
    tag: 'Stage 2',
    model: 'QLoRA Rank 16, Alpha 16 (3 Epochs, 1,314 Steps)',
    sampleScope: 'Not evaluated (no standalone SFT eval artifact)',
    executionAccuracy: null,
    sqlValidity: null,
    executionSuccess: null,
    description: 'Supervised fine-tuning on 7,000 Spider training pairs; standalone held-out eval artifact was not saved',
    keyMilestone: 'SFT adapter trained (1.18% params); standalone eval pending Kaggle GPU run',
  },
  {
    stageId: 3,
    name: 'RLVR Policy (Greedy Pass@1)',
    tag: 'Stage 3',
    model: 'GRPO Policy (100 Steps, G=8, 4 NB3 Rewards)',
    sampleScope: 'first_50 (N=50, 2 DBs); 59.0% EX on dev[:100]',
    executionAccuracy: 66.0,
    sqlValidity: 78.0,
    executionSuccess: 78.0,
    description: '66.0% EX (33/50) on audited first_50; 59.0% EX (+4.0 pp vs Zero-Shot) on matched dev[:100] (N=100)',
    keyMilestone: '+4.0 pp vs Zero-Shot on matched dev[:100] (59.0% vs 55.0%)',
  },
  {
    stageId: 4,
    name: 'Majority@8 Consensus Voting',
    tag: 'Stage 4',
    model: 'Test-Time Search (T=0.8, 8 Samples, Multiset Hash)',
    sampleScope: 'Spider first_50 (N=50, 2 DBs)',
    executionAccuracy: 80.0,
    sqlValidity: 96.0,
    executionSuccess: 96.0,
    description: 'Plurality voting over 8 stochastic candidates with multiset hash clustering on first_50 (N=50)',
    keyMilestone: '+14.0 pp over Greedy (first_50); 90.24% precision on consensus (37/41)',
  },
];

export const BENCHMARK_SUMMARY = {
  benchmarkName: 'Spider 1.0 (Audited Dev Slice: first_50)',
  evalMode: 'first_50',
  nExamples: 50,
  evalDatabasesNote: '2 databases: 45 concert_singer, 5 pets_1 (full 1,034-example run pending)',
  adapterSha256: '9EF9C5326FC68CE4D9D602392F3DE107877C2C8083BC77A74D3B68799409ECF3',
  percentileMethod: 'linear (NumPy default, Hyndman & Fan Type 7)',
  matched100Slice: {
    nExamples: 100,
    zeroShotExPct: 55.0,
    grpoGreedyExPct: 59.0,
    deltaPts: 4.0,
  },
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
    selfCorrectionRatePct: 16.0,
    regressionRatePct: 2.0,
    decisionStatus: {
      CONSENSUS_ELECTED: {
        count: 41,
        pct: 82.0,
        accuracy: 90.24,
        meaning: 'Plurality consensus (K >= 3 identical multiset hashes). Safe for autonomous execution (37/41 correct).',
      },
      LOW_CONFIDENCE: {
        count: 6,
        pct: 12.0,
        accuracy: 50.0,
        meaning: 'Winning cluster has K < 3 support (NB4 scored winning SQL 3/6; Serving API abstains for review).',
      },
      NO_CONSENSUS_TIE: {
        count: 1,
        pct: 2.0,
        accuracy: 0.0,
        meaning: 'Tie between competing candidate clusters (NB4 scored first-seen 0/1; Serving API abstains).',
      },
      ALL_FAILED: {
        count: 2,
        pct: 4.0,
        accuracy: 0.0,
        meaning: 'All 8 sampled candidates encountered AST rejection or runtime execution errors (0/2; abstain).',
      },
    },
  },
};

export const REWARD_FUNCTIONS_INFO = [
  {
    name: 'R_fmt (Format Validity)',
    bounds: '[-0.50, +0.20]',
    weight: 'Strict Enclosure',
    description: 'Enforces strict <sql>...</sql> tag structure without trailing text.',
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
    description: 'Ensures SQL executes cleanly in the deterministic SQLite sandbox (2.0s package / 3.0s NB3 deadline).',
  },
  {
    name: 'R_result (Multiset Accuracy)',
    bounds: '[0.00, +1.00]',
    weight: 'Ground Truth Verification',
    description: 'Order-independent multiset equivalence match against verified ground truth results.',
  },
  {
    name: 'R_eff (Latency Efficiency)',
    bounds: '[0.00, +0.30]',
    weight: 'Gated (Package Only)',
    description: 'Correctness-gated latency bonus in package composite reward (not active in NB3 4-reward training).',
  },
];
