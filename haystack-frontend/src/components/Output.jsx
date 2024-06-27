import React, { useState } from 'react';
import './Output.css';

function Output({ activeButton, imageData }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');
    const [showKeywordCount, setShowKeywordCount] = useState(false);
    const [showConceptFreq, setShowConceptFreq] = useState(false);
    const [showTopicModel, setShowTopicModel] = useState(false);

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = (e) => {
        e.preventDefault();
        // For now, echo the question as the response
        // In a real application, you'd likely call an API here
        setResponse(`You asked: "${question}"`);
    };

    const toggleKeywordCount = () => {
        setShowKeywordCount(!showKeywordCount);
    };

    const toggleConceptFreq = () => {
        setShowConceptFreq(!showConceptFreq);
    };

    const toggleTopicModel = () => {
        setShowTopicModel(!showTopicModel);
    };

    return (
        <div className="output-container">
            {activeButton === 'left' && (
                <>
                    <div className="left-section">
                    <h2>Generated Plot</h2>
                    {imageData && (
                        <img src={`data:image/png;base64,${imageData}`} alt="Generated Plot" />
                    )}
                    </div>
                </>
                
            ) && (
                <>
                    <div className='left-section'>
                        <div className="expandable-section">
                            <button onClick={toggleKeywordCount} className="expand-button">
                                {showKeywordCount ? 'Hide Plot Details' : 'Show Plot Details'}
                            </button>
                            {showKeywordCount && (
                                <div className="keyword-count">
                                    {/* Include plot details here */}
                                    <p>Plot details content goes here...</p>
                                </div>
                            )}
                        </div>
                        <div className="expandable-section">
                            <button onClick={toggleConceptFreq} className="expand-button">
                                {showConceptFreq ? 'Hide Statistics' : 'Show Statistics'}
                            </button>
                            {showConceptFreq && (
                                <div className="concept-freq">
                                    {/* Include statistics content here */}
                                    <p>Statistics content goes here...</p>
                                </div>
                            )}
                        </div>
                        <div className="expandable-section">
                            <button onClick={toggleTopicModel} className="expand-button">
                                {showTopicModel ? 'Hide Explanation' : 'Show Explanation'}
                            </button>
                            {showTopicModel && (
                                <div className="topic-model">
                                    {/* Include explanation content here */}
                                    <p>Explanation content goes here...</p>
                                </div>
                            )}
                        </div>
                    </div>
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
