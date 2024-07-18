import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, generateFileHash, getAllPdfs, deletePdfById, clearDatabase } from '../utils/indexedDB';
import { auth } from '../firebase/config';
import { useAuth } from '../contexts/AuthContext';
import './Upload.css';

const MAX_TOTAL_SIZE = 200 * 1024 * 1024; // 20MB

function Upload({ loading, pdfList, onFileChange, onCheckboxChange, onPdfRemove }) {
    const [dragging, setDragging] = useState(false);
    const [pdfsTotalSize, setPdfsTotalSize] = useState(0);
    const [uploadStatus, setUploadStatus] = useState({}); 
    const [uploading, setUploading] = useState(false); 
    const hashList = [];
    const { userLoggedIn } = useAuth();

    // useEffect(() => {
    //     const clearData = async () => {
    //         await clearDatabase();
    //         onFileChange([]); 
    //         setUploadStatus({});
    //         setPdfsTotalSize(0);
    //     };

    //     clearData();
    // }, []);

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
        await uploadFiles(files);
    };

    // const removeDuplicates = (pdfList) => {
    //     const pdfs = pdfList.filter((pdf, index, self) => index === self.findIndex((t) => (t.hash === pdf.hash)));
    //     console.log(pdfs);
    //     return pdfs;
    // };

    const handleFileInputChange = async (event) => {
        //const user = auth.currentUser;
        if (!userLoggedIn) {
            alert('You are not authenticated. Please sign in to upload PDFs.');
            console.error('User not authenticated');
            return;
        }

        //const user_id = user.uid;
        const formData = new FormData();
        const newfiles = [];

        const files = Array.from(event.target.files);
        if (!await checkTotalSize(files)) {
            return;
        }

        await uploadFiles(files);
    };

    const uploadFiles = async (files) => {
        //const user = auth.currentUser;
        // if (!userLoggedIn) {
        //     console.error('User not authenticated');
        //     return;
        // }

        //const user_id = user.uid;
        const formData = new FormData();
        const newFiles = [];
        const allPdfs = await getAllPdfs();

        setUploading(true);

        for (let file of files) {
            const hash = await generateFileHash(file);

            if (hashList.includes(hash) || allPdfs.some(pdf => pdf.hash === hash)) {
                alert(`${file.name} is a duplicate. Skipping...`);
                continue;
            } else {
                hashList.push(hash);
                newFiles.push(file);
                await addPdfToDatabase(file);
                setUploadStatus(prev => ({ ...prev, [file.name]: 'loading' }));
            }
        }

        newFiles.forEach((file) => {
            formData.append('pdf_list', file);
            formData.append('doc_ids', file.name);
        });
        //formData.append('user_id', user_id);

        // Debugging: Log FormData entries
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }

        setUploading(true);

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
                newFiles.forEach(file => {
                    setUploadStatus(prev => ({ ...prev, [file.name]: 'completed' }));
                });
            } else {
                const errorData = await response.json();
                console.error('Error:', errorData.message);
                newFiles.forEach(file => {
                    setUploadStatus(prev => ({ ...prev, [file.name]: 'failed' }));
                });
            }
        } catch (error) {
            console.error('Error:', error.message);
            newFiles.forEach(file => {
                setUploadStatus(prev => ({ ...prev, [file.name]: 'failed' }));
            });
        } finally {
            // Check file size and add to database
            if (!await checkTotalSize(files)) {
                return;
            }
            setUploading(false);
            onFileChange();
        }
    };

    const handleRemovePdf = (id) => {
        const pdf = pdfList.find((pdf) => pdf.id === id);
        if (pdf) {
            deletePdfById(id);
            onPdfRemove(id);
            setUploadStatus(prev => {
                const updatedStatus = { ...prev };
                delete updatedStatus[pdf.name];
                return updatedStatus;
            });
        }
    };

    return (
        <div className="upload-pdfs">
            <h1 id="title">Upload your PDFs</h1>
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
                    style={{ display: 'none' }}
                />
                <label className="file-input-label" htmlFor="file-input">
                    Browse
                </label>
            </div>

            <div className="uploaded-files">
                {pdfList.length === 0 && <p>No PDFs uploaded yet</p>}
                {pdfList.length > 0 && !uploading && (
                    <>
                        <p>Uploaded PDFs:</p>
                        <div className="pdf-list">
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
                                            onClick={() => handleRemovePdf(pdf.id)}
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

            {uploading && (
                <div className="upload-progress">
                    {Object.keys(uploadStatus).map((fileName) => (
                        <div key={fileName} className="upload-progress-item">
                            <p>{fileName}</p>
                            {uploadStatus[fileName] === 'loading' ? (
                                <div className="loading-animation"></div>
                            ) : (
                                <div className="check-mark">&#10003;</div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default Upload;