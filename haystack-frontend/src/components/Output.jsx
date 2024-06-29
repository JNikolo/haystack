import React, { useState } from 'react';
import NlpButtons from './NlpButtons';
import KeywordCounting from './KeywordCount';
import ConceptFreq from './ConceptFreq';
import './Output.css';
import TopicModeling from './TopicModeling';

function Output({ activeButton, pdfList }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');
    const [activeButtonNlp, setActiveButtonNlp] = useState('left');
    const [buttonClicked, setButtonClicked] = useState(false);


    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = (e) => {
        e.preventDefault();
        // For now, echo the question as the response
        // In a real application, you'd likely call an API here
        setResponse(`You asked: "${question}"`);
    };

    

    const handleButtonClick = (nlpType) => {
        setActiveButtonNlp(nlpType);
        setButtonClicked(true);
    };



    return (
        <div className="output-container">
            {activeButton === 'left' && (
                <>
                    <NlpButtons
                        activeButtonNlp={activeButtonNlp}
                        buttonClicked={buttonClicked}
                        onButtonClick={handleButtonClick}
                    />
                    <div className='nlpOutput'>
                        {activeButtonNlp === 'left' && (
                            <KeywordCounting
                                pdfList={pdfList}
                            />
                        )}
                        {activeButtonNlp === 'middle' && (
                            <ConceptFreq
                            pdfList={pdfList}
                            />
                        )}
                        {activeButtonNlp === 'right' && (
                            <TopicModeling />
                        )}
                    </div>
                </>
            )}
            {/* // )} && <NlpOptions activeButton={activeButtonNlp} setActiveButton={setActiveButtonNlp} />} */}
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
