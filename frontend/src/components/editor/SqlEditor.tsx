import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check, Download, Wand2, Terminal } from 'lucide-react';

interface SqlEditorProps {
  value: string;
  onChange?: (val: string) => void;
  readOnly?: boolean;
  statusBadge?: React.ReactNode;
}

export const SqlEditor: React.FC<SqlEditorProps> = ({
  value,
  onChange,
  readOnly = false,
  statusBadge,
}) => {
  const [copied, setCopied] = useState(false);
  const [internalValue, setInternalValue] = useState(value);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(internalValue);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = internalValue;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([internalValue], { type: 'text/sql' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_${Date.now()}.sql`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleFormat = () => {
    // Client-side SQL format cleaner (standardize keywords to uppercase and tidy indentation)
    const keywords = [
      'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN',
      'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN', 'ON',
      'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'ALL',
      'AS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IS', 'NULL', 'DESC', 'ASC'
    ];

    let formatted = internalValue;
    // Normalize keywords
    keywords.forEach(kw => {
      const regex = new RegExp(`\\b${kw}\\b`, 'gi');
      formatted = formatted.replace(regex, kw);
    });

    setInternalValue(formatted);
    onChange?.(formatted);
  };

  const lines = internalValue.split('\n');

  return (
    <div
      style={{
        border: '1px solid var(--qa-border)',
        borderRadius: 'var(--qa-radius-md)',
        backgroundColor: 'var(--qa-bg-card)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Editor Header / Toolbar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 14px',
          backgroundColor: 'var(--qa-bg-elevated)',
          borderBottom: '1px solid var(--qa-border)',
          fontSize: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={14} color="var(--qa-accent)" />
          <span style={{ fontWeight: 600, color: 'var(--qa-text-primary)' }}>
            SQL Editor
          </span>
          <span
            style={{
              fontSize: '10px',
              color: 'var(--qa-text-muted)',
              fontFamily: 'var(--qa-font-mono)',
              backgroundColor: 'var(--qa-bg-card)',
              padding: '2px 6px',
              borderRadius: '4px',
              border: '1px solid var(--qa-border)',
            }}
          >
            SQLite 3.x Dialect
          </span>
          {statusBadge}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {!readOnly && (
            <button
              onClick={handleFormat}
              className="qa-btn-ghost"
              title="Format SQL Keywords"
              style={{ fontSize: '11px', padding: '4px 8px' }}
            >
              <Wand2 size={13} />
              Format
            </button>
          )}

          <button
            onClick={handleCopy}
            className="qa-btn-ghost"
            title="Copy SQL to Clipboard"
            style={{ fontSize: '11px', padding: '4px 8px' }}
          >
            {copied ? (
              <>
                <Check size={13} color="var(--qa-success)" />
                <span style={{ color: 'var(--qa-success)' }}>Copied</span>
              </>
            ) : (
              <>
                <Copy size={13} />
                Copy
              </>
            )}
          </button>

          <button
            onClick={handleDownload}
            className="qa-btn-ghost"
            title="Download SQL File (.sql)"
            style={{ fontSize: '11px', padding: '4px 8px' }}
          >
            <Download size={13} />
            Export
          </button>
        </div>
      </div>

      {/* Editor Content with Line Numbers & Input */}
      <div style={{ display: 'flex', minHeight: '140px', position: 'relative' }}>
        {/* Line Numbers Column */}
        <div
          style={{
            userSelect: 'none',
            backgroundColor: 'var(--qa-bg-elevated)',
            color: 'var(--qa-text-muted)',
            fontFamily: 'var(--qa-font-mono)',
            fontSize: '13px',
            lineHeight: '1.6',
            padding: '12px 10px',
            textAlign: 'right',
            borderRight: '1px solid var(--qa-border)',
            minWidth: '42px',
          }}
        >
          {lines.map((_, i) => (
            <div key={i}>{i + 1}</div>
          ))}
        </div>

        {/* Text Area Input */}
        <textarea
          ref={textareaRef}
          value={internalValue}
          readOnly={readOnly}
          onChange={e => {
            setInternalValue(e.target.value);
            onChange?.(e.target.value);
          }}
          spellCheck={false}
          style={{
            flex: 1,
            backgroundColor: 'transparent',
            color: 'var(--qa-text-primary)',
            fontFamily: 'var(--qa-font-mono)',
            fontSize: '13px',
            lineHeight: '1.6',
            padding: '12px 14px',
            border: 'none',
            outline: 'none',
            resize: 'vertical',
            width: '100%',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        />
      </div>

      {/* Editor Footer Status */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '6px 14px',
          backgroundColor: 'var(--qa-bg-elevated)',
          borderTop: '1px solid var(--qa-border-subtle)',
          fontSize: '11px',
          color: 'var(--qa-text-muted)',
          fontFamily: 'var(--qa-font-mono)',
        }}
      >
        <span>
          {lines.length} {lines.length === 1 ? 'line' : 'lines'}, {internalValue.length} chars
        </span>
        <span>{readOnly ? 'Locked (Model Output)' : 'Editable'}</span>
      </div>
    </div>
  );
};
