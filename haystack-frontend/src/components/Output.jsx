import React, { useEffect, useState } from 'react';
import { auth } from '../firebase/config'; // Ensure this path is correct
import NlpButtons from './NlpButtons';
import KeywordCounting from './KeywordCount';
import ConceptFreq from './ConceptFreq';
import './Output.css';
import TopicModeling from './TopicModeling';

function Output({ activeButton, pdfList }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState(null);
    const [activeButtonNlp, setActiveButtonNlp] = useState('left');
    const [buttonClicked, setButtonClicked] = useState(false);

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = async (e) => {
        e.preventDefault();

        // Get the current user
        const user = auth.currentUser;
        if (!user) {
            setResponse('Error: User not authenticated');
            return;
        }

        const user_id = user.uid; // Get the current user's UUID
        const doc_ids = pdfList.map(pdf => pdf.file.name); // Assuming pdfList contains objects with an 'id' field

        const requestBody = {
            query: question,
            user_id: user_id,
            doc_ids: doc_ids
        };

        try {
            const response = await fetch('http://localhost:8000/qa_rag/', { // Replace with your backend URL
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (response.ok) {
                const data = await response.json();
                setResponse(data);
            } else {
                const errorData = await response.json();
                setResponse({ error: errorData.detail });
            }
        } catch (error) {
            setResponse({ error: error.message });
        }
    };

    const handleButtonClick = (nlpType) => {
        setActiveButtonNlp(nlpType);
        setButtonClicked(true);
    };

    const renderResponse = () => {
        if (!response) {
            return null;
        }
        if (response.error) {
            return <p>Error: {response.error}</p>;
        }
        if (Array.isArray(response)) {
            return response.map((reply, index) => (
                <div key={index}>
                    <h3>Response {index + 1}:</h3>
                    <pre>{JSON.stringify(reply, null, 2)}</pre>
                </div>
            ));
        }
        return (
            <div>
                <h3>Response:</h3>
                <pre>{JSON.stringify(response, null, 2)}</pre>
            </div>
        );
    };

    useEffect(() => {
        const handleBeforeUnload = async (event) => {
            // Prevent the default action
            event.preventDefault();
            event.returnValue = '';

            const user = auth.currentUser;
            if (!user) {
                console.error('User not authenticated');
                return;
            }

            const user_id = user.uid;

            // Make the fetch call to your backend
            try {
                const response = await fetch(`http://localhost:8000/delete_embeddings/?user_id=${user_id}`, {
                    method: 'DELETE',
                    // Add any headers or body data if required
                });

                if (!response.ok) {
                    console.error('Error deleting embeddings');
                }
            } catch (error) {
                console.error('Error:', error);
            }
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        // Cleanup the event listener on component unmount
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

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
                    <div className="response">
                        {renderResponse()}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Output;
