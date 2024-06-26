import React, { useState } from 'react';
import './Output.css';

function Output({ activeButton, imageData }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = (e) => {
        e.preventDefault();
        // For now, echo the question as the response
        // In a real application, you'd likely call an API here
        setResponse(`You asked: "${question}"`);
    };

    return (
        <div className="output-container">
            {activeButton === 'left' && imageData && (
                <>
                    <h2>Generated Plot</h2>
                    <img src={`data:image/png;base64,${imageData}`} alt="Generated Plot" />
                </>
            )}
            {activeButton === 'right' && (
                <div className="question-section">
                    <h2>Ask a Question</h2>
                    <form onSubmit={handleQuestionSubmit} className="question-form">
                        <textarea
                            value={question}
                            onChange={handleQuestionChange}
                            placeholder="Type your question here"
                            className="question-input"
                            rows={1} 
                        />
                        <button type="submit" className="question-submit">Ask</button>
                    </form>
                    {response && (
                        <div className="response">
                            <h3>Response:</h3>
                            <p>{response}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default Output;
