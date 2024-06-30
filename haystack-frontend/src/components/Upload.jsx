import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, getAllPdfs, deletePdfById } from '../utils/indexedDB';
import './Upload.css';

function Upload({ loading, pdfList, onFileChange, onCheckboxChange, onPdfRemove }) {
    // const [pdfs, setPdfs] = useState([]);
    // //const [submitted, setSubmitted] = useState(false);

    // useEffect(() => {
    //     loadPdfs();
    // }, []);

    // const loadPdfs = async () => {
    //     const allPdfs = await getAllPdfs();
    //     setPdfs(allPdfs);
    // };

    // const handleFileChange = async (event) => {
    //     const files = Array.from(event.target.files);
    //     for (let file of files) {
    //         await addPdfToDatabase(file);
    //     }
    //     loadPdfs();
    // };

    // // const handleSubmitClick = async () => {
    // //     setSubmitted(true);
    // //     await handleSubmit(pdfs.filter(pdf => pdf.selected));
    // // };

    // // const handleCheckboxChange = (id) => {
    // //     setPdfs(pdfs.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf));
    // // };

    // const handleCheckboxChange = (id) => {
    //     const updatedPdfs = pdfs.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf);
    //     setPdfs(updatedPdfs);
    //     onPdfRemoved(updatedPdfs);
    // };


    // const handleRemovePdf = async (id) => {
    //     await deletePdfById(id);
    //     const updatedPdfs = pdfs.filter(pdf => pdf.id !== id);
    //     setPdfs(updatedPdfs);

    //     // Notify parent component (GetInsights) about the updated pdfList
    //     onPdfRemoved(updatedPdfs);
    // };

    const [dragging, setDragging] = useState(false);

    const handleDragEnter = (event) => {
        event.preventDefault();
        setDragging(true);
    };

    const handleDragLeave = (event) => {
        event.preventDefault();
        setDragging(false);
    };

    const handleDragOver = (event) => {
        event.preventDefault();
        setDragging(true);
    };

    const handleDrop = async (event) => {
        event.preventDefault();
        setDragging(false);

        const files = Array.from(event.dataTransfer.files);
        for (let file of files) {
            await addPdfToDatabase(file);
        }
        onFileChange();
    };

    const handleFileInputChange = async (event) => {
        const files = Array.from(event.target.files);
        for (let file of files) {
            await addPdfToDatabase(file);
        }
        onFileChange();
    };

    return (
        <div className="upload-pdfs">
            <h1 id="title">Upload your PDFs</h1>
            {/* Drag and Drop Area */}
            <div
                className={`file-drop-area ${dragging ? 'dragging' : ''}`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                <h3>Drag & Drop files here</h3>
                <p>Limit 200MB in total</p>
                <p>Only .pdf accepted</p>
            </div>

            {/* File Input for Browse Button */}
            <input
                type="file"
                multiple
                accept="application/pdf"
                id="file-input"
                onChange={handleFileInputChange}
                style={{ display: 'none' }} // Hide the file input visually
            />
            <label className="file-input-label" htmlFor="file-input">
                Browse
            </label>
            <div className="uploaded-files">
                {pdfList.length === 0 && <p>No PDFs uploaded yet</p>}
                {pdfList.length > 0 && (
                    <>
                        <p>Uploaded PDFs:</p>
                        <div className='pdf-list'>
                            <ul>
                                {pdfList.map((pdf) => (
                                    <li key={pdf.id} className="pdf-item">
                                        <input
                                            type="checkbox"
                                            id={`checkbox-${pdf.id}`}
                                            checked={pdf.selected || false}
                                            onChange={() => onCheckboxChange(pdf.id)}
                                        />
                                        <label htmlFor={`checkbox-${pdf.id}`}>{pdf.name}</label>
                                        <button
                                            className="remove-button"
                                            onClick={() => onPdfRemove(pdf.id)}
                                            disabled={loading}
                                        >
                                            X
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

export default Upload;