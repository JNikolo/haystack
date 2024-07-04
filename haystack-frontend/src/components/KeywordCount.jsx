import React, { useState } from 'react';
import './KeywordCount.css';
import { getPdfById } from '../utils/indexedDB';

function KeywordCounting({ selectedPdfs }) {
    const [keyword, setKeyword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [apiResponse, setApiResponse] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);

        // Filter selected PDFs
        //const selectedPdfs = pdfList.filter(pdf => pdf.selected);

        // Create a FormData object to send the selected PDFs

        const formData = new FormData();
        // for (let pdf of selectedPdfs) {
        //     formData.append('files', pdf.file);
        // }

        for (let pdfID of selectedPdfs) {
            const pdf = await getPdfById(pdfID);
            formData.append('files', pdf.file);
        }



        try {
            const response = await fetch(`http://127.0.0.1:8000/searchkeyword/?keyword=${keyword}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to fetch keyword count');
            }

            const data = await response.json();
            if (data.status === 'success') {
                setApiResponse(data);
                setError(null); // Reset error state if successful
            } else {
                throw new Error(data.message || 'Failed to fetch keyword count');
            }
        } catch (error) {
            setError(error.message || 'Failed to fetch keyword count');
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeywordChange = (e) => {
        setKeyword(e.target.value);
    };

    // Reset apiResponse and error when user clicks submit again
    const handleNewSearch = () => {
        setKeyword('');
        setApiResponse(null);
        setError(null);
    };

    const downloadResults = () => {
        // Function to handle download of results in JSON format
        if (apiResponse) {
            const jsonData = JSON.stringify(apiResponse, null, 2);
            const blob = new Blob([jsonData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `keyword_results.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    };

    return (
        <div className="keyword-counting">
            <h2>Get keyword counting for your PDFs!</h2>
            <form onSubmit={handleSubmit} className="keyword-form">
                <textarea
                    value={keyword}
                    onChange={handleKeywordChange}
                    placeholder="Type your keyword here"
                    className="keyword-input"
                    rows={1}
                    disabled={isLoading || !(selectedPdfs.length > 0)} // Disable textarea when loading or no PDFs selected
                />
            </form>
            <button type="submit" className="keyword-submit" disabled={isLoading || !(selectedPdfs.length > 0)}>
                    {isLoading ? 'Counting...' : 'Submit'}
            </button>
            {error && <p className="error-message">{error}</p>}
            {apiResponse && (
                <div className="keyword-output">
                    <p>{`The keyword "${keyword}" appeared ${apiResponse.total} times.`}</p>
                    <button className="download-button" onClick={downloadResults}>
                        Download Results (JSON)
                    </button>
                </div>
            )}
            {/* Render a button to clear results */}
            {apiResponse && (
                <button className="new-search-button" onClick={handleNewSearch}>
                    New Search
                </button>
            )}
        </div>
    );
}

export default KeywordCounting;
