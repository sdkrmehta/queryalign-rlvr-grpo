import React, { useState, useEffect, useCallback } from 'react';
import {
  ActivePage,
  DatabaseMetadata,
  InferenceMode,
  TextToSQLResponse,
} from './types';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { WorkspacePage } from './pages/WorkspacePage';
import { HistoryPage } from './pages/HistoryPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { SchemaExplorerPage } from './pages/SchemaExplorerPage';
import { SettingsPage } from './pages/SettingsPage';
import {
  checkBackendHealth,
  fetchDatabases,
  fetchSchema,
  generateTextToSql,
} from './services/api';
import {
  getStoredHistory,
  getStoredMode,
  getStoredTheme,
  saveHistoryItem,
  setStoredTheme,
} from './services/storage';

export const App: React.FC = () => {
  // Navigation & Theme State
  const [activePage, setActivePage] = useState<ActivePage>('workspace');
  const [theme, setThemeState] = useState<'dark' | 'light'>(() => getStoredTheme());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Backend & Database State
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);
  const [databases, setDatabases] = useState<string[]>(['test_company']);
  const [selectedDatabase, setSelectedDatabase] = useState<string>('test_company');
  const [schemaMetadata, setSchemaMetadata] = useState<DatabaseMetadata | null>(null);
  const [schemaPromptText, setSchemaPromptText] = useState<string>('');

  // Workspace & Query State
  const [question, setQuestion] = useState<string>('list employee name');
  const [mode, setMode] = useState<InferenceMode>(() => getStoredMode());
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [lastResponse, setLastResponse] = useState<TextToSQLResponse | null>(null);
  const [editedSql, setEditedSql] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [historyCount, setHistoryCount] = useState<number>(() => getStoredHistory().length);

  // Apply theme to document element
  const handleSetTheme = (newTheme: 'dark' | 'light') => {
    setThemeState(newTheme);
    setStoredTheme(newTheme);
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Health Check Poller
  const checkHealth = useCallback(async () => {
    try {
      await checkBackendHealth();
      setIsBackendOnline(true);
    } catch {
      setIsBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Load Database List on Mount
  useEffect(() => {
    async function loadDatabases() {
      try {
        const dbs = await fetchDatabases();
        if (dbs && dbs.length > 0) {
          setDatabases(dbs);
          if (!dbs.includes(selectedDatabase)) {
            setSelectedDatabase(dbs[0]);
          }
        }
      } catch (err) {
        console.warn('Could not fetch databases list on mount:', err);
      }
    }
    loadDatabases();
  }, [selectedDatabase]);

  // Load Schema when selected database changes
  useEffect(() => {
    async function loadSchema() {
      if (!selectedDatabase) return;
      try {
        const schemaRes = await fetchSchema(selectedDatabase);
        setSchemaMetadata(schemaRes.metadata);
        setSchemaPromptText(schemaRes.schema_text);
      } catch (err) {
        console.warn(`Could not load schema for database "${selectedDatabase}":`, err);
        setSchemaMetadata(null);
        setSchemaPromptText('');
      }
    }
    loadSchema();
  }, [selectedDatabase]);

  // Refresh history count badge
  const refreshHistoryBadge = useCallback(() => {
    setHistoryCount(getStoredHistory().length);
  }, []);

  // Generate & Execute Query Handler
  const handleGenerate = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    setError(null);

    const numCandidates = mode === 'majority8' ? 8 : 1;

    try {
      const response = await generateTextToSql({
        question: question.trim(),
        database_id: selectedDatabase,
        num_candidates: numCandidates,
        execute: true,
      });

      setLastResponse(response);
      setEditedSql(response.sql || '');

      // Persist to local storage history
      saveHistoryItem({
        databaseId: selectedDatabase,
        question: question.trim(),
        sql: response.sql || undefined,
        status: (response.status as any) || 'success',
        mode,
        rowCount: response.execution?.row_count,
        latencyMs: response.latency_ms?.total || response.latency_ms?.sandbox,
      });
      refreshHistoryBadge();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown generation error occurred.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion('');
    setLastResponse(null);
    setEditedSql('');
    setError(null);
  };

  // Cross-page navigation handlers
  const handleLoadQueryIntoWorkspace = (
    loadedQuestion: string,
    loadedDb: string,
    loadedMode: InferenceMode
  ) => {
    setQuestion(loadedQuestion);
    setSelectedDatabase(loadedDb);
    setMode(loadedMode);
    setActivePage('workspace');
  };

  const handleNavigateToWorkspaceWithQuery = (
    initialQuestion: string,
    targetDb: string
  ) => {
    setQuestion(initialQuestion);
    setSelectedDatabase(targetDb);
    setActivePage('workspace');
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', width: '100vw', backgroundColor: 'var(--qa-bg-primary)' }}>
      {/* Collapsible Left Sidebar */}
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        historyCount={historyCount}
      />

      {/* Main Layout Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflowX: 'hidden' }}>
        {/* Top Sticky Header */}
        <Header
          activePage={activePage}
          theme={theme}
          setTheme={handleSetTheme}
          isBackendOnline={isBackendOnline}
          selectedDatabase={selectedDatabase}
        />

        {/* Page Content Container */}
        <main style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
          {activePage === 'workspace' && (
            <WorkspacePage
              question={question}
              setQuestion={setQuestion}
              selectedDatabase={selectedDatabase}
              setSelectedDatabase={setSelectedDatabase}
              databases={databases}
              mode={mode}
              setMode={setMode}
              onGenerate={handleGenerate}
              isLoading={isLoading}
              onClear={handleClear}
              schemaMetadata={schemaMetadata}
              schemaPromptText={schemaPromptText}
              lastResponse={lastResponse}
              editedSql={editedSql}
              setEditedSql={setEditedSql}
              error={error}
            />
          )}

          {activePage === 'history' && (
            <HistoryPage
              onLoadQueryIntoWorkspace={handleLoadQueryIntoWorkspace}
              onRefreshHistoryBadge={refreshHistoryBadge}
            />
          )}

          {activePage === 'evaluation' && <EvaluationPage />}

          {activePage === 'model-info' && <ModelInfoPage />}

          {activePage === 'schema-explorer' && (
            <SchemaExplorerPage
              databases={databases}
              selectedDatabase={selectedDatabase}
              setSelectedDatabase={setSelectedDatabase}
              metadata={schemaMetadata}
              schemaPromptText={schemaPromptText}
              onNavigateToWorkspaceWithQuery={handleNavigateToWorkspaceWithQuery}
            />
          )}

          {activePage === 'settings' && (
            <SettingsPage
              theme={theme}
              setTheme={handleSetTheme}
              defaultMode={mode}
              setDefaultMode={setMode}
              onRefreshHistoryBadge={refreshHistoryBadge}
            />
          )}
        </main>
      </div>
    </div>
  );
};
