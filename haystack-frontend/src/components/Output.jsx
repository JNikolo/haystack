import React, { useEffect, useState } from 'react';
import { auth } from '../firebase/config'; // Ensure this path is correct
import NlpButtons from './NlpButtons';
import KeywordCounting from './KeywordCount';
import ConceptFreq from './ConceptFreq';
import './Output.css';
import TopicModeling from './TopicModeling';
import { getPdfById } from '../utils/indexedDB';
import Loading from './Loading';
import parse from 'html-react-parser';
import DOMPurify from 'dompurify';

function Output({ activeButton }) {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [keyword, setKeyword] = useState('');
    const [error, setError] = useState(null);
    // const [activeButtonNlp, setActiveButtonNlp] = useState('left');
    // const [buttonClicked, setButtonClicked] = useState(true);
    useEffect(() => {
        const response = JSON.parse(localStorage.getItem('llm-response'));
        if (response) {
            setResponse(response);
        }
        else{
            setResponse(null);
        }
    }, []);

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleQuestionSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        const newpdfList = [];

        const selectedPdfs = JSON.parse(localStorage.getItem('selectedPdfs'));

        if (!selectedPdfs || selectedPdfs.length === 0 ) {
            alert('Please select PDFs');
            setIsLoading(false);
            return;
        }
        if (question === ''){
            alert('Please type your question!');
            setIsLoading(false);
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
            if (pdf) {
                newpdfList.push(pdf);
            }
            else{
                console.log('PDF not found in indexedDB');
                const filteredPdfs = selectedPdfs.filter(pdf => pdf !== pdfID);
                localStorage.setItem('selectedPdfs', JSON.stringify(filteredPdfs));
                setIsLoading(false);
                setError('A PDF was not found! Try again.');
                return;
            }
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
                localStorage.setItem('llm-response', JSON.stringify(data));
            } else {
                const errorData = await response.json();
                setResponse({ error: errorData.detail });
            }
        } catch (error) {
            setResponse({ error: error.message });
        } finally {
            setIsLoading(false);
        }
    };

    // const handleButtonClick = async (nlpType) => {
    //     await setActiveButtonNlp(nlpType);
    //     setButtonClicked(true);
    // };    

    const renderResponse = () => {
        if (!response) {
            return null;
        }
        if (error) {
            return <p>Error: {error}</p>;
        }
        if (response.error) {
            return <p>Error: {response.error}</p>;
        }
        if (response) {
            console.log('doc_ids: ', response.doc_ids);
            console.log('answers: ', response.result);
            
            let answers = response.result;
            answers = answers.map(answer => DOMPurify.sanitize(answer));
            let doc_ids = response.doc_ids;
            if (Array.isArray(answers) && Array.isArray(doc_ids) && answers.length === doc_ids.length) {
                return answers.map((answer, index) => (
                    <div className='qa_output' key={index}>
                        <div>
                            <h3><strong>{`${doc_ids[index]}`}</strong></h3>
                            <p>Answer:</p>
                            <div className='html-parsed'>{parse(answer)}</div>
                        </div>
                    </div>
                ));
            }
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
            {/* {activeButton === 'home' && (
                <>
                    <Home />
                </>
            )} */}
            {activeButton === 'keyword_count' && (
                <>
                    <KeywordCounting keyword={keyword} setKeyword={setKeyword}/>
                </>
            )}
            {activeButton === 'ner' && (
                <>
                    <ConceptFreq />
                </>
            )}
            {activeButton === 'topic_modeling' && (
                <>
                    <TopicModeling />
                </>
            )}
            {activeButton === 'q_a' && (
                <>
                    <div className="question-section">
                        <h2>Query your documents!</h2>
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
                            
                            {isLoading && (
                                <div className="response-loading">
                                    <Loading />
                                </div> 
                            )}
                            {!isLoading && renderResponse()}
                        </div>
                    </div>
                </>
            )}
            


            {/* {activeButton === 'left' && (
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
            )} */}
        </div>
    );
}

export default Output;
