import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, getAllPdfs, deletePdfById } from '../utils/indexedDB';
import './Upload.css';

function Upload({ loading, onPdfRemoved, onDeletePdf }) {
    const [pdfs, setPdfs] = useState([]);
    //const [submitted, setSubmitted] = useState(false);

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

    // const handleSubmitClick = async () => {
    //     setSubmitted(true);
    //     await handleSubmit(pdfs.filter(pdf => pdf.selected));
    // };

    // const handleCheckboxChange = (id) => {
    //     setPdfs(pdfs.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf));
    // };

    const handleCheckboxChange = (id) => {
        const updatedPdfs = pdfs.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf);
        setPdfs(updatedPdfs);
        onPdfRemoved(updatedPdfs);
    };


    const handleRemovePdf = async (id) => {
        await deletePdfById(id);
        const updatedPdfs = pdfs.filter(pdf => pdf.id !== id);
        setPdfs(updatedPdfs);

        // Notify parent component (GetInsights) about the updated pdfList
        onPdfRemoved(updatedPdfs);
    };

    return (
        <div className="App">
            <h1 id="title">Upload PDFs and Generate Plot</h1>
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
                        <li key={pdf.id} className="pdf-item">
                            <input
                                type="checkbox"
                                id={`checkbox-${pdf.id}`}
                                checked={pdf.selected || false}
                                onChange={() => handleCheckboxChange(pdf.id)}
                            />
                            <label htmlFor={`checkbox-${pdf.id}`}>{pdf.name}</label>
                            <button
                                className="remove-button"
                                onClick={() => handleRemovePdf(pdf.id)}
                                disabled={loading}
                            >
                                X
                            </button>
                        </li>
                ))}
            </ul>
        </div>
    );
}

export default Upload;
