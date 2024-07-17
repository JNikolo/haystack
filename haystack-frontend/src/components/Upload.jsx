import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, generateFileHash, getAllPdfs, deletePdfById} from '../utils/indexedDB';
import { auth } from '../firebase/config';
import './Upload.css';

const MAX_TOTAL_SIZE = 200 * 1024 * 1024; // 20MB

function Upload({ loading, pdfList, onFileChange, onCheckboxChange, onPdfRemove }) {
    const [dragging, setDragging] = useState(false);
    const [pdfsTotalSize, setPdfsTotalSize] = useState(0);
    const hashList = [];

    const checkTotalSize = async (files) => {
        const totalSize = files.reduce((acc, file) => acc + file.size, 0);
        if (totalSize + pdfsTotalSize > MAX_TOTAL_SIZE) {
            alert('Total size of PDFs exceeds 20MB');
            return false;
        }
        setPdfsTotalSize(totalSize + pdfsTotalSize);
        return true;
    };

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

        const allPdfs = files.every(file => file.type === 'application/pdf');
        if (!allPdfs) {
            alert('Only PDF files are accepted.');
            return;
        }

        if (!await checkTotalSize(files)) {
            return;
        }
        for (let file of files) {
            await addPdfToDatabase(file);
        }
        onFileChange();
    };

    // const removeDuplicates = (pdfList) => {
    //     const pdfs = pdfList.filter((pdf, index, self) => index === self.findIndex((t) => (t.hash === pdf.hash)));
    //     console.log(pdfs);
    //     return pdfs;
    // };

    const handleFileInputChange = async (event) => {
        const user = auth.currentUser;
        if (!user) {
            console.error('User not authenticated');
            return;
        }

        const user_id = user.uid;
        const formData = new FormData();
        const newfiles = [];

        const files = Array.from(event.target.files);
        if (!await checkTotalSize(files)) {
            return;
        }

        const allPdfs = await getAllPdfs();
        for (let file of files) {
            const hash = await generateFileHash(file);
            
            if (hashList.includes(hash) || allPdfs.some(pdf => pdf.hash === hash)){
                alert(`${file.name} is a duplicate. Skipping...`);
                continue;
            }
            else{
                hashList.push(hash);
                newfiles.push(file);
                await addPdfToDatabase(file);
            }
        }

        newfiles.forEach((file) => {
            formData.append('pdf_list', file);
            formData.append('doc_ids', file.name); // Assuming doc_id is just the index for this example
        });
        //formData.append('user_id', user_id);

        // Debugging: Log FormData entries
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/add_embeddings/', { // Replace with your backend URL
                method: 'POST',
                mode: 'cors', 
                body: formData,
                credentials: 'include',
            });

            if (response.ok) {
                const data = await response.json();
                console.log('Success:', data.message);
                // Perform any additional actions needed upon success
            } else {
                const errorData = await response.json();
                console.error('Error:', errorData.message);
                // Handle the error accordingly
            }
        } catch (error) {
            console.error('Error:', error.message);
            // Handle the error accordingly
        }

        // Check file size and add to database
        if (!await checkTotalSize(files)) {
            return;
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
                <p>Drag & Drop files here</p>
                <p>Limit 200MB in total</p>
                <p>Only .pdf accepted</p>

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
            </div>

            {/* File Input for Browse Button */}
            
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
                                            onChange={(event) => onCheckboxChange(event, pdf.id)}
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