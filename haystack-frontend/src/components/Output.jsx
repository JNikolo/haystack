import React, { useEffect, useState } from 'react';
import { auth } from '../firebase/config'; // Ensure this path is correct
import NlpButtons from './NlpButtons';
import KeywordCounting from './KeywordCount';
import ConceptFreq from './ConceptFreq';
import './Output.css';
import TopicModeling from './TopicModeling';
import { getPdfById } from '../utils/indexedDB';


function Output({ activeButton }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');
    const [activeButtonNlp, setActiveButtonNlp] = useState('left');
    const [buttonClicked, setButtonClicked] = useState(true);

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = async (e) => {
        e.preventDefault();
        const newpdfList = [];

        const selectedPdfs = JSON.parse(localStorage.getItem('selectedPdfs'));

        if (!selectedPdfs || selectedPdfs.length === 0 ) {
            alert('Please select PDFs');
            //setIsLoading(false);
            return;
        }
        if (question === ''){
            alert('Please type your question!');
            //setIsLoading(false);
            return;
        }
        
        // Get the current user
        const user = auth.currentUser;
        if (!user) {
            setResponse('Error: User not authenticated');
            return;
        }

        for (let pdfID of selectedPdfs) {
            const pdf = await getPdfById(pdfID);
            newpdfList.push(pdf);
        }

        //const user_id = user.uid; // Get the current user's UUID
        const doc_ids = newpdfList.map(pdf => pdf.name); // Assuming pdfList contains objects with an 'id' field

        //console.log('Query: ', question);
        //console.log('User ID: ', user_id);
        //console.log('Doc IDS: ', doc_ids);

        const requestBody = {
            query: question,
            //user_id: user_id,
            doc_ids: doc_ids
        };

        console.log('Request Body: ', requestBody);

        try {
            const response = await fetch('http://127.0.0.1:8000/qa_rag/', { // Replace with your backend URL
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                mode: 'cors',
                credentials: 'include',
                body: JSON.stringify(requestBody)
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

    const handleButtonClick = async (nlpType) => {
        await setActiveButtonNlp(nlpType);
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

    // useEffect(() => {
    //     const handleBeforeUnload = async (event) => {
    //         // Prevent the default action
    //         event.preventDefault();
    //         event.returnValue = '';

    //         const user = auth.currentUser;
    //         if (!user) {
    //             console.error('User not authenticated');
    //             return;
    //         }

    //         //const user_id = user.uid;

    //         // Make the fetch call to your backend
    //         try {
    //             const response = await fetch('http://127.0.0.1:8000/delete_embeddings/', {
    //                 method: 'DELETE',
    //                 credentials: 'include',
    //                 mode: 'cors',
    //                 // Add any headers or body data if required
    //             });

    //             if (!response.ok) {
    //                 console.error('Error deleting embeddings');
    //             }
    //         } catch (error) {
    //             console.error('Error:', error);
    //         }
    //     };

    //     window.addEventListener('beforeunload', handleBeforeUnload);

    //     // Cleanup the event listener on component unmount
    //     return () => {
    //         window.removeEventListener('beforeunload', handleBeforeUnload);
    //     };
    // }, []);

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
                            <KeywordCounting />
                        )}
                        {activeButtonNlp === 'middle' && (
                            <ConceptFreq />
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
