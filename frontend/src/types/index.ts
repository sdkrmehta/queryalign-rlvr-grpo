// ============================================================================
// QueryAlign Backend API & Domain Type Definitions
// ============================================================================

export type InferenceMode = 'greedy' | 'majority8';

export interface ValidationStatus {
  format_valid: boolean;
  ast_safe: boolean;
  error_code?: string | null;
  error_message?: string | null;
}

export interface ExecutionStatus {
  success: boolean;
  complete: boolean;
  row_count: number;
  truncated: boolean;
  error_code?: string | null;
  error_message?: string | null;
}

export interface ConsensusStatus {
  enabled: boolean;
  cluster_size: number;
  total_candidates: number;
  confidence_score: number;
  abstention_reason?: string | null;
}

export interface TextToSQLRequest {
  database_id: string;
  question: string;
  num_candidates?: number;
  execute?: boolean;
}

export interface TextToSQLResponse {
  request_id: string;
  status: 'success' | 'format_error' | 'ast_rejected' | string;
  sql?: string | null;
  validation: ValidationStatus;
  execution?: ExecutionStatus | null;
  result_preview?: Array<Array<string | number | null>> | null;
  columns?: string[] | null;
  consensus?: ConsensusStatus | null;
  latency_ms: {
    total?: number;
    sandbox?: number;
    [key: string]: number | undefined;
  };
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface DatabasesResponse {
  databases: string[];
}

export interface ColumnInfo {
  name: string;
  type: string;
  is_pk: boolean;
  samples?: string[];
}

export interface TableInfo {
  columns: ColumnInfo[];
  primary_keys: string[];
}

export interface ForeignKeyInfo {
  from_table: string;
  from_column: string;
  to_table: string;
  to_column: string;
}

export interface DatabaseMetadata {
  tables: Record<string, TableInfo>;
  foreign_keys: ForeignKeyInfo[];
}

export interface SchemaResponse {
  database_id: string;
  schema_text: string;
  metadata: DatabaseMetadata;
}

// Local UI & History Models
export interface QueryHistoryItem {
  id: string;
  timestamp: number;
  databaseId: string;
  question: string;
  sql?: string;
  status: 'success' | 'format_error' | 'ast_rejected' | 'error';
  mode: InferenceMode;
  rowCount?: number;
  latencyMs?: number;
  isFavorite?: boolean;
}

export type ActivePage =
  | 'workspace'
  | 'history'
  | 'evaluation'
  | 'model-info'
  | 'schema-explorer'
  | 'settings';
