import React, { useState, useEffect } from 'react';
import { addPdfToDatabase, generateFileHash, getAllPdfs, deletePdfById, clearDatabase } from '../utils/indexedDB';
import { auth } from '../firebase/config';
import { useAuth } from '../contexts/AuthContext';
import './Upload.css';
import { IoMdCloudUpload, IoMdClose } from "react-icons/io";

const MAX_TOTAL_SIZE = 200 * 1024 * 1024; // 200MB

function Upload({ loading, pdfList, onFileChange, onCheckboxChange, onPdfRemove }) {
    const [dragging, setDragging] = useState(false);
    //const [pdfsTotalSize, setPdfsTotalSize] = useState(0);
    const [uploadStatus, setUploadStatus] = useState({}); 
    const [uploading, setUploading] = useState(false);
    const [hashList, setHashList] = useState([]);
    const { userLoggedIn } = useAuth();

    useEffect(() => {

        const initializeHashList = async () => {
            const allPdfs = await getAllPdfs();
            const hashes = allPdfs.map(pdf => pdf.hash);
            setHashList(hashes);
        };

        initializeHashList();
    }, []);

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
        let pdfsTotalSize = localStorage.getItem('pdfsTotalSize');
        pdfsTotalSize = pdfsTotalSize ? parseInt(pdfsTotalSize, 10) : 0;

        console.log('current total size: ', pdfsTotalSize);


        const totalSize = files.reduce((acc, file) => acc + file.size, 0);

        console.log('New pdfs total size: ', totalSize);

        const newTotalSize = totalSize + pdfsTotalSize;
        if (newTotalSize > MAX_TOTAL_SIZE) {
            alert('Total size of PDFs exceeds 200MB');
            return false;
        }
        //setPdfsTotalSize(totalSize + pdfsTotalSize);

        console.log('New total size: ', newTotalSize);

        localStorage.setItem('pdfsTotalSize', newTotalSize);
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

        const isAllPdfs = files.every(file => file.type === 'application/pdf');

        const preProcessedFiles = [];
        if (!isAllPdfs) {
            alert('Only PDF files are accepted.');
            return;
        }

        // if (!await checkTotalSize(files)) {
        //     return;
        // }

        for (let file of files) {
            const hash = await generateFileHash(file);
            console.log(`PDF ${file.name} hash is: ${hash}`);

            if (hashList.includes(hash)) {
                console.log("Skipping file...");
                alert(`${file.name} is already uploaded. Skipping file...`);
                continue;
            } else {
                console.log("New file, adding to database");
                setHashList([...hashList, hash]);
                preProcessedFiles.push(file);
                await addPdfToDatabase(file);
                setUploadStatus(prev => ({ ...prev, [file.name]: 'loading' }));
            }
        }

        if (preProcessedFiles.length > 0){
            if (!await checkTotalSize(preProcessedFiles)) {
                return;
            }

            await uploadFiles(preProcessedFiles);
        }
        else {
            console.log('No pdfs after preprocessing :(');
            return;
        }
        // onFileChange();
        // await uploadFiles(files);
    };

    // const removeDuplicates = (pdfList) => {
    //     const pdfs = pdfList.filter((pdf, index, self) => index === self.findIndex((t) => (t.hash === pdf.hash)));
    //     console.log(pdfs);
    //     return pdfs;
    // };

    const handleFileInputChange = async (event) => {
        //const user = auth.currentUser;
        console.log('Trying to upload a pdf starting');
        if (!userLoggedIn) {
            alert('You are not authenticated. Please sign in to upload PDFs.');
            console.error('User not authenticated');
            return;
        }

        //const user_id = user.uid;
        // const formData = new FormData();
        // const newfiles = [];

        console.log('These files are going to be uploaded');

        const files = Array.from(event.target.files);
        const preProcessedFiles = [];
        console.log('Files to be uploaded:', files);

        console.log("Hash List: ", hashList);

        for (let file of files) {
            const hash = await generateFileHash(file);
            console.log(`PDF ${file.name} hash is: ${hash}`);

            if (hashList.includes(hash)) {
                console.log("Skipping file...");
                alert(`${file.name} is already uploaded. Skipping file...`);
                continue;
            } else {
                console.log("New file, adding to database");
                setHashList([...hashList, hash]);
                preProcessedFiles.push(file);
                await addPdfToDatabase(file);
                setUploadStatus(prev => ({ ...prev, [file.name]: 'loading' }));
            }
        }


        if (preProcessedFiles.length > 0){
            if (!await checkTotalSize(preProcessedFiles)) {
                return;
            }

            await uploadFiles(preProcessedFiles);
        }
        else {
            console.log('No pdfs after preprocessing :(');
            return;
        }

        // Clear the input value to ensure handleFileInputChange is called on re-upload
        event.target.value = ''; // Clear file input after upload
    };

    const uploadFiles = async (files) => {
        //const user = auth.currentUser;
        // if (!userLoggedIn) {
        //     console.error('User not authenticated');
        //     return;
        // }

        //somehow fixed the bug where the pdf was not being uploaded

        // if (files.length === 1 && pdfList.length === 1){
        //     console.log('Entering if statement for bug fix');
        //     const file = files[0];
        //     const hash = await generateFileHash(file);
        //     console.log(`PDF ${file.name} hash is: ${hash}`);
        //     if (hashList.includes(hash)) {
        //         console.log("Skipping file...");
        //         alert(`${file.name} is already uploaded. Skipping file...`);
        //     } else {
        //         console.log("New file, adding to database");
        //         setHashList([...hashList, hash]);
        //         preProcessedFiles.push(file);
        //         await addPdfToDatabase(file);
        //         setUploadStatus(prev => ({ ...prev, [file.name]: 'loading' }));
        //     }
        // }
        //const user_id = user.uid;
        const formData = new FormData();
        //const newFiles = [];
        //const allPdfs = await getAllPdfs();

        setUploading(true);

        

        files.forEach((file) => {
            formData.append('pdf_list', file);
            formData.append('doc_ids', file.name);
        });
        //formData.append('user_id', user_id);

        // Debugging: Log FormData entries
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }

        //setUploading(true);

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
                files.forEach(file => {
                    setUploadStatus(prev => ({ ...prev, [file.name]: 'completed' }));
                });
            } else {
                const errorData = await response.json();
                console.error('Error:', errorData.message);
                files.forEach(file => {
                    setUploadStatus(prev => ({ ...prev, [file.name]: 'failed' }));
                });
            }
        } catch (error) {
            console.error('Error:', error.message);
            files.forEach(file => {
                setUploadStatus(prev => ({ ...prev, [file.name]: 'failed' }));
            });
        } finally {
            // Check file size and add to database
            // if (!await checkTotalSize(files)) {
            //     return;
            // }
            setUploading(false);
            onFileChange();
        }
    };

    const handleRemovePdf = (id) => {
        const pdf = pdfList.find((pdf) => pdf.id === id);
        if (pdf) {
            //deletePdfById(id);
            onPdfRemove(id);


            //remove pdf size from total size
            let pdfsTotalSize = parseInt(localStorage.getItem('pdfsTotalSize'), 10);
            console.log('pdfsTotalSize: ', pdfsTotalSize);
            console.log(pdf);
            console.log('pdf size: ', pdf.file.size);

            pdfsTotalSize -= pdf.file.size;

            localStorage.setItem('pdfsTotalSize', pdfsTotalSize.toString());

            // Update hashList after removing PDF
            setHashList(prevHashList => prevHashList.filter(hash => hash !== pdf.hash));

            setUploadStatus(prev => {
                const updatedStatus = { ...prev };
                delete updatedStatus[pdf.name];
                return updatedStatus;
            });
        }
    };

    const handleDeleteAll = async () => {

        if (!userLoggedIn) {
            alert('You are not authenticated. Please sign in to upload PDFs.');
            console.error('User not authenticated');
            return;
        }

        try{
            
            // Call delete_embeddings route after signing out
            const response = await fetch('http://127.0.0.1:8000/delete_embeddings/', {
                method: 'DELETE',
                credentials: 'include',
                mode: 'cors',
                    // Add any headers or body data if required
            });
        
            if (response.ok) {
                await clearDatabase(); // Clear the IndexedDB database
                localStorage.removeItem('selectedPdfs');
                localStorage.removeItem('pdfsTotalSize');
                localStorage.removeItem('plotData');
                localStorage.removeItem('llm-response');
                localStorage.removeItem('topicData');
                console.log("Embeddings deleted successfully");
            } else {
                console.error("Failed to delete embeddings:", response.statusText);
            }
        } catch (error) {
            console.error("Error signing out:", error);
        } finally {
            setUploadStatus({});
            setHashList([]);
            onFileChange();
        }
    }

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
                <div className='upload-text-container'>
                    <div className="upload-icon"><IoMdCloudUpload size={30}/></div>
                    <div className='upload-text'>
                        <h4>Drag & Drop files here</h4>
                        <p>Limit: 200MB in total!</p>
                    </div>
                </div>

                <div className='browse-button-container'>
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
            </div>

            <div className="uploaded-files">
                {pdfList.length === 0 && <h1>No PDFs uploaded yet</h1>}
                {pdfList.length > 0 && !uploading && (
                    <>
                        <h1 className='uploaded-pdf-text'>Uploaded PDFs:</h1>
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
                                            <IoMdClose size={13}/>
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        <div className="delete-button-container">
                            <button
                                className="delete-button"
                                onClick={handleDeleteAll}
                                disabled={uploading}
                            >
                                Delete All
                            </button>
                        </div>
                    </>
                )}
            </div>

            {uploading && (
                <div className="pdf-list">
                    {Object.keys(uploadStatus).map((fileName) => (
                        <div key={fileName} className="upload-progress-item">
                            <div className='filename'><p>{fileName}</p></div>
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