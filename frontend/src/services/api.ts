import {
  DatabasesResponse,
  HealthResponse,
  SchemaResponse,
  TextToSQLRequest,
  TextToSQLResponse,
} from '../types';

const API_BASE = ''; // Uses Vite proxy in development; relative URL in production

export class ApiError extends Error {
  constructor(public statusCode: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Health check returned HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new Error('FastAPI serving backend is unreachable (check port 8000).');
  }
}

export async function fetchDatabases(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/databases`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Database listing returned HTTP ${res.status}`);
    }
    const data: DatabasesResponse = await res.json();
    return data.databases || [];
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new Error('Unable to fetch approved database registry from backend.');
  }
}

export async function fetchSchema(databaseId: string): Promise<SchemaResponse> {
  try {
    const res = await fetch(`${API_BASE}/databases/${encodeURIComponent(databaseId)}/schema`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Schema fetch failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new Error(`Failed to load schema metadata for database "${databaseId}".`);
  }
}

export async function generateTextToSql(request: TextToSQLRequest): Promise<TextToSQLResponse> {
  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const errorText = await res.text();
      let errorDetail = `Generation request failed with HTTP ${res.status}`;
      try {
        const parsed = JSON.parse(errorText);
        if (parsed.detail) errorDetail = parsed.detail;
      } catch {
        if (errorText) errorDetail = errorText;
      }
      throw new ApiError(res.status, errorDetail);
    }
    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new Error(err instanceof Error ? err.message : 'Text-to-SQL generation failed.');
  }
}
