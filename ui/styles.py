"""Gradioアプリで使用するスタイル。"""

CUSTOM_CSS = """
.header {
    margin-bottom: 30px;
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.header h1 {
    margin-bottom: 10px;
    font-size: 2.5rem;
    font-weight: 700;
}

.header p {
    margin: 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

.config-card,
.results-card {
    margin-bottom: 25px;
    padding: 25px;
    border: 1px solid #e1e8ed;
    border-radius: 15px;
    background: white;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.config-card h3,
.results-card h3 {
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #3498db;
    color: #2c3e50;
    font-weight: 600;
}

.button-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
    padding: 30px 0;
}

.primary-btn,
.secondary-btn {
    border: 0;
    color: white;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.primary-btn {
    min-width: 280px;
    padding: 18px 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    font-size: 1.2rem;
    font-weight: 600;
}

.secondary-btn {
    width: 100%;
    margin: 10px 0;
    padding: 12px 24px;
    border-radius: 8px;
    background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
    font-size: 1rem;
    font-weight: 500;
}

.primary-btn:hover,
.secondary-btn:hover {
    transform: translateY(-2px);
}

.primary-btn:hover {
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.secondary-btn:hover {
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.status-success,
.status-error,
.status-warning {
    margin: 15px 0;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-weight: 600;
}

.status-success {
    background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
}

.status-error {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
}

.status-warning {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.metric-card {
    padding: 20px;
    border-left: 4px solid #667eea;
    border-radius: 12px;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.metric-value {
    margin-bottom: 5px;
    color: #2c3e50;
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    color: #6c757d;
    font-size: 0.9rem;
    font-weight: 500;
}

.agent-result {
    overflow: hidden;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    background: #f8f9fa;
}

.agent-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 20px;
    color: white;
    background: linear-gradient(135deg, #495057 0%, #343a40 100%);
    cursor: pointer;
    font-weight: 600;
}

.agent-content {
    padding: 20px;
    border-top: 1px solid #dee2e6;
}

.agent-content pre {
    padding: 15px;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    background: white;
    white-space: pre-wrap;
    font-family: "Courier New", monospace;
    font-size: 0.9rem;
    line-height: 1.5;
}

@keyframes fade-in {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fade-in 0.6s ease-out;
}

@media (max-width: 768px) {
    .header h1 {
        font-size: 2rem;
    }

    .header p {
        font-size: 1rem;
    }

    .config-card,
    .results-card {
        padding: 15px;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }
}
"""
