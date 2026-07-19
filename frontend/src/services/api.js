import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = {
  // Public configuration
  getConfig: () => {
    return axios.get(`${API_BASE_URL}/api/config`);
  },

  // Session management
  createSession: () => {
    return axios.post(`${API_BASE_URL}/api/session/create`);
  },

  getSession: (sessionId) => {
    return axios.get(`${API_BASE_URL}/api/session/${sessionId}`);
  },

  deleteSession: (sessionId) => {
    return axios.delete(`${API_BASE_URL}/api/session/${sessionId}`);
  },

  getHistory: (sessionId, limit = 50) => {
    return axios.get(`${API_BASE_URL}/api/session/${sessionId}/history`, {
      params: { limit }
    });
  },

  // Document management
  uploadDocument: (file, sessionId) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    return axios.post(`${API_BASE_URL}/api/documents/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params: {
        session_id: sessionId
      },
      timeout: 600000  // 10 minutes - embedding large docs with rate-limit throttling takes time
    });
  },

  deleteDocument: (sessionId, documentId) => {
    return axios.delete(`${API_BASE_URL}/api/session/${sessionId}/documents/${documentId}`);
  },

  // Query (non-streaming)
  query: (question, sessionId) => {
    return axios.post(`${API_BASE_URL}/api/query`, {
      question,
      session_id: sessionId
    });
  },

  // Query (streaming via SSE)
  queryStream: (question, sessionId, onToken, onSources, onDone, onError) => {
    const controller = new AbortController();

    fetch(`${API_BASE_URL}/api/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal: controller.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          const error = await response.json();
          onError(error.detail || 'Stream request failed');
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event = JSON.parse(line.slice(6));

                if (event.type === 'token') {
                  onToken(event.content);
                } else if (event.type === 'sources') {
                  onSources(event.sources);
                } else if (event.type === 'done') {
                  onDone(event.token_usage);
                } else if (event.type === 'error') {
                  onError(event.content);
                }
              } catch (e) {
                // Skip malformed events
              }
            }
          }
        }
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          onError(err.message);
        }
      });

    // Return abort function for cleanup
    return () => controller.abort();
  }
};

export default api;
