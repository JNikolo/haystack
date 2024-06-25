import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, getAllPdfs } from '../utils/indexedDB';
import './Upload.css';

function Upload({ loading, handleSubmit }) {
    const [pdfs, setPdfs] = useState([]);

    useEffect(() => {
        loadPdfs();
    }, []);

    const loadPdfs = async () => {
        const allPdfs = await getAllPdfs();
        setPdfs(allPdfs);
    };

    const handleFileChange = async (event) => {
        const files = Array.from(event.target.files);
        for (let file of files) {
            await addPdfToDatabase(file);
        }
        loadPdfs();
    };

    const handleCheckboxChange = (id) => {
        setPdfs(pdfs.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf));
    };

    return (
        <div className="App">
            <h1>Upload PDFs and Generate Plot</h1>
            <input
                type="file"
                multiple
                accept="application/pdf"
                id="file-input"
                onChange={handleFileChange}
            />
            <label className="file-input-label" htmlFor="file-input">
                Browse
            </label>
            <ul>
                {pdfs.map((pdf) => (
                    <li key={pdf.id}>
                        <input
                            type="checkbox"
                            id={`checkbox-${pdf.id}`}
                            checked={pdf.selected || false}
                            onChange={() => handleCheckboxChange(pdf.id)}
                        />
                        <label htmlFor={`checkbox-${pdf.id}`}>{pdf.name}</label>
                    </li>
                ))}
            </ul>
            {pdfs.length > 0 && (
                <button
                    className="submit-button"
                    onClick={() => handleSubmit(pdfs.filter(pdf => pdf.selected))}
                    disabled={loading}
                >
                    {loading ? 'Uploading...' : 'Submit'}
                </button>
            )}
        </div>
    );
}

export default Upload;




