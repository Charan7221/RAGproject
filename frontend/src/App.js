import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import api from './services/api';

const SESSION_STORAGE_KEY = 'rag_session_id';
const CHAT_HISTORY_KEY = 'rag_chat_histories'; // Store all document chats

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [totalChunks, setTotalChunks] = useState(0);
  const [chatHistories, setChatHistories] = useState({}); // { documentId: [messages] }
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [config, setConfig] = useState(null);

  const welcomeMessage = {
    role: 'assistant',
    content: 'Hello! Upload a document to get started. I\'ll answer questions based on the selected document.',
    timestamp: new Date().toISOString()
  };

  // Load chat histories from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(CHAT_HISTORY_KEY);
    if (saved) {
      try {
        setChatHistories(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading chat histories:', e);
      }
    }
  }, []);

  // Save chat histories to localStorage whenever they change
  useEffect(() => {
    if (Object.keys(chatHistories).length > 0) {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistories));
    }
  }, [chatHistories]);

  const restoreSession = async (storedSessionId) => {
    try {
      const sessionResponse = await api.getSession(storedSessionId);
      const { documents: docs, total_chunks } = sessionResponse.data;

      setSessionId(storedSessionId);
      setDocuments(docs);
      setTotalChunks(total_chunks);

      // Auto-select first document if available
      if (docs.length > 0 && !selectedDocumentId) {
        setSelectedDocumentId(docs[0].id);
      }

      // Load chat history from API if available
      try {
        const historyResponse = await api.getHistory(storedSessionId);
        const history = historyResponse.data.history || [];
        
        // Try to organize history by document (this is a fallback for existing data)
        if (history.length > 0 && docs.length > 0) {
          // If we have history but no document-specific storage, put it in first doc
          const firstDocId = docs[0].id;
          if (!chatHistories[firstDocId] || chatHistories[firstDocId].length === 0) {
            setChatHistories(prev => ({
              ...prev,
              [firstDocId]: history.map((msg) => ({
                role: msg.role,
                content: msg.content,
                timestamp: new Date().toISOString()
              }))
            }));
          }
        }
      } catch (error) {
        console.error('Error loading history:', error);
      }
    } catch (error) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      await createNewSession();
    }
  };

  const createNewSession = async () => {
    const response = await api.createSession();
    const newSessionId = response.data.session_id;
    localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
    setSessionId(newSessionId);
    setDocuments([]);
    setTotalChunks(0);
    setSelectedDocumentId(null);
    setChatHistories({});
  };

  // Initialize or restore session on mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const configResponse = await api.getConfig();
        setConfig(configResponse.data);

        const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
        if (storedSessionId) {
          await restoreSession(storedSessionId);
        } else {
          await createNewSession();
        }
      } catch (error) {
        console.error('Error initializing session:', error);
      }
    };

    initSession();
  }, []);

  // Handle document upload
  const handleUpload = async (files) => {
    setLoading(true);
    
    try {
      const uploadedDocs = [];
      
      for (const file of files) {
        const response = await api.uploadDocument(file, sessionId);
        const doc = response.data.document;
        uploadedDocs.push(doc);
        setTotalChunks(response.data.total_chunks);
        
        // Initialize chat for this document
        setChatHistories(prev => ({
          ...prev,
          [doc.id]: [{
            role: 'assistant',
            content: `Document "${doc.filename}" uploaded successfully! Ask me anything about this document.`,
            timestamp: new Date().toISOString()
          }]
        }));
      }
      
      setDocuments(prev => [...prev, ...uploadedDocs]);
      
      // Auto-select the first uploaded document
      if (uploadedDocs.length > 0 && !selectedDocumentId) {
        setSelectedDocumentId(uploadedDocs[0].id);
      }
      
    } catch (error) {
      console.error('Error uploading document:', error);
      alert(`Error uploading document: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle sending query (with streaming) - only queries selected document
  const handleSendMessage = async (question) => {
    if (!question.trim() || !selectedDocumentId) return;
    
    // Add user message to selected document's chat
    const userMessage = {
      role: 'user',
      content: question,
      timestamp: new Date().toISOString()
    };
    
    setChatHistories(prev => ({
      ...prev,
      [selectedDocumentId]: [...(prev[selectedDocumentId] || []), userMessage]
    }));
    
    setLoading(true);
    setIsStreaming(true);
    setStreamingContent('');
    
    let fullContent = '';
    let streamSources = [];
    
    // Use streaming API
    const abort = api.queryStream(
      question,
      sessionId,
      // onToken
      (token) => {
        fullContent += token;
        setStreamingContent(fullContent);
      },
      // onSources
      (sources) => {
        streamSources = sources;
      },
      // onDone
      (tokenUsage) => {
        // Add completed assistant message to selected document's chat
        const assistantMessage = {
          role: 'assistant',
          content: fullContent,
          sources: streamSources,
          token_usage: tokenUsage,
          timestamp: new Date().toISOString()
        };
        
        setChatHistories(prev => ({
          ...prev,
          [selectedDocumentId]: [...(prev[selectedDocumentId] || []), assistantMessage]
        }));
        
        setStreamingContent('');
        setIsStreaming(false);
        setLoading(false);
      },
      // onError
      (error) => {
        console.error('Stream error:', error);
        const errorMessage = {
          role: 'system',
          content: `❌ Error: ${error}`,
          timestamp: new Date().toISOString()
        };
        
        setChatHistories(prev => ({
          ...prev,
          [selectedDocumentId]: [...(prev[selectedDocumentId] || []), errorMessage]
        }));
        
        setStreamingContent('');
        setIsStreaming(false);
        setLoading(false);
      }
    );
    
    return abort;
  };

  // Handle document deletion - also delete its chat
  const handleDeleteDocument = async (documentId) => {
    try {
      await api.deleteDocument(sessionId, documentId);
      
      // Remove document
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
      
      // Remove its chat history
      setChatHistories(prev => {
        const newHistories = { ...prev };
        delete newHistories[documentId];
        return newHistories;
      });
      
      // If deleted document was selected, select another one
      if (selectedDocumentId === documentId) {
        const remainingDocs = documents.filter(doc => doc.id !== documentId);
        setSelectedDocumentId(remainingDocs.length > 0 ? remainingDocs[0].id : null);
      }
      
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('Error deleting document');
    }
  };

  // Handle clear all
  const handleClearAll = async () => {
    if (window.confirm('Clear all documents and chat history?')) {
      try {
        await api.deleteSession(sessionId);
        localStorage.removeItem(SESSION_STORAGE_KEY);
        localStorage.removeItem(CHAT_HISTORY_KEY);
        await createNewSession();
      } catch (error) {
        console.error('Error clearing session:', error);
      }
    }
  };

  // Get current messages for selected document
  const currentMessages = selectedDocumentId && chatHistories[selectedDocumentId] 
    ? chatHistories[selectedDocumentId] 
    : [];

  // Get selected document info
  const selectedDocument = documents.find(doc => doc.id === selectedDocumentId);

  return (
    <div className="App">
      <Sidebar
        documents={documents}
        selectedDocumentId={selectedDocumentId}
        onSelectDocument={setSelectedDocumentId}
        totalChunks={totalChunks}
        config={config}
        onUpload={handleUpload}
        onDeleteDocument={handleDeleteDocument}
        onClearAll={handleClearAll}
        loading={loading}
      />
      <ChatInterface
        messages={currentMessages}
        selectedDocument={selectedDocument}
        onSendMessage={handleSendMessage}
        loading={loading}
        hasDocuments={documents.length > 0}
        streamingContent={streamingContent}
        isStreaming={isStreaming}
      />
    </div>
  );
}

export default App;
