import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './Sidebar.css';

function Sidebar({ documents, selectedDocumentId, onSelectDocument, totalChunks, config, onUpload, onDeleteDocument, onClearAll, loading }) {
  
  const onDrop = useCallback((acceptedFiles) => {
    onUpload(acceptedFiles);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    multiple: true
  });

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>📁 Document Manager</h2>
      </div>

      <div className="sidebar-content">
        {/* Settings Section */}
        <div className="sidebar-section">
          <h3>⚙️ Settings</h3>
          <div className="settings-info">
            <div className="setting-item">
              <span className="setting-label">Model:</span>
              <span className="setting-value">{config?.llm_model || 'Loading...'}</span>
            </div>
            <div className="setting-item">
              <span className="setting-label">Top K:</span>
              <span className="setting-value">{config?.top_k ?? '...'}</span>
            </div>
            {config?.hybrid_search_enabled && (
              <div className="setting-item">
                <span className="setting-label">Search:</span>
                <span className="setting-value">Hybrid (RRF)</span>
              </div>
            )}
          </div>
        </div>

        {/* Upload Section */}
        <div className="sidebar-section">
          <h3>📤 Upload Documents</h3>
          <div 
            {...getRootProps()} 
            className={`dropzone ${isDragActive ? 'active' : ''} ${loading ? 'disabled' : ''}`}
          >
            <input {...getInputProps()} disabled={loading} />
            {isDragActive ? (
              <p>Drop files here...</p>
            ) : (
              <>
                <p>Drag & drop files here</p>
                <p className="dropzone-hint">or click to browse</p>
                <p className="dropzone-formats">PDF, DOCX, TXT, MD</p>
              </>
            )}
          </div>
        </div>

        {/* Documents List */}
        <div className="sidebar-section">
          <h3>📚 Your Documents ({documents.length})</h3>
          {documents.length === 0 ? (
            <p className="empty-state">No documents uploaded</p>
          ) : (
            <div className="documents-list">
              {documents.map((doc) => (
                <div 
                  key={doc.id} 
                  className={`document-item ${selectedDocumentId === doc.id ? 'selected' : ''}`}
                  onClick={() => onSelectDocument(doc.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="document-info">
                    <div className="document-name">
                      {selectedDocumentId === doc.id && <span className="selected-icon">📄 </span>}
                      {selectedDocumentId !== doc.id && <span className="check-icon">✓ </span>}
                      {doc.filename}
                    </div>
                    <div className="document-meta">
                      {formatFileSize(doc.size)} • {doc.chunks} chunks
                    </div>
                  </div>
                  <button
                    className="delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm(`Delete "${doc.filename}" and its chat history?`)) {
                        onDeleteDocument(doc.id);
                      }
                    }}
                    title="Delete document and chat"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* System Status */}
        <div className="sidebar-section">
          <h3>📊 System Status</h3>
          <div className="status-info">
            <div className="status-item">
              <span className="status-indicator success">●</span>
              System Ready
            </div>
            <div className="status-item">
              <span className="status-label">Total Chunks:</span>
              <span className="status-value">{totalChunks}</span>
            </div>
          </div>
        </div>

        {/* Actions */}
        {documents.length > 0 && (
          <div className="sidebar-actions">
            <button
              className="clear-button"
              onClick={onClearAll}
              disabled={loading}
            >
              🗑️ Clear All
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default Sidebar;

