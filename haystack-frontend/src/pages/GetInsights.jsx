import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
//import Upload from "../components/Upload";
import Output from "../components/Output";
import Options from "../components/Options";
import Upload from "../components/Upload";
import './GetInsights.css';
import { clearDatabase, getAllPdfs, deletePdfById, addPdfToDatabase } from '../utils/indexedDB';

function GetInsights() {
    const [loading, setLoading] = useState(false);
    const [activeButton, setActiveButton] = useState('keyword_count');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [pdfList, setPdfList] = useState([]);


    useEffect(() => {
        const handleBeforeUnload = (event) => {
            const message = "Are you sure you want to leave? Unsaved changes may be lost.";
            event.preventDefault(); // Standard way to display a prompt in some browsers
            event.returnValue = message; // For others
            return message;
        };

        const handleUnload = () => {
            clearDatabase().then(() => {
                console.log("Database cleared on session end");
            }).catch((error) => {
                console.error("Error clearing the database: ", error);
            });
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        window.addEventListener('unload', handleUnload);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            window.removeEventListener('unload', handleUnload);
        };
    }, []);

    useEffect(() => {
        loadPdfs();
    }
    , []);

    // const handleSubmit = async (selectedPdfs) => {
    //     setLoading(true);

    //     // const formData = new FormData();
      
    //     let newPdfList = [];
    //     for (let pdf of selectedPdfs) {
    //         const pdfRecord = await getPdfById(pdf.id);
    //         newPdfList.push(pdfRecord);
    //     }

    //     setPdfList(newPdfList);

    //     // call the API to populate vector db

    //     setLoading(false);
            
    //     //     formData.append('files', pdfRecord.file);
    //     //     console.log('pdf: ',formData);
    //     //console.log(pdfList);
    // };

    // const handleReset = async () => {
    //     setLoading(true);
    //     // await clearDatabase();
    //     setPdfList([]);

    //     // call API here to delete all vector DB records
        
    //     setLoading(false);
    // };
    
    // const handleDeletePdf = async (id) => {
    //     await deletePdfById(id);
    //     const updatedPdfs = pdfList.filter(pdf => pdf.id !== id);
    //     setPdfList(updatedPdfs);
    // };

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    const loadPdfs = async () => {
        const allPdfs = await getAllPdfs();
        setPdfList(allPdfs);
    };

    // const handleFileChange = async (event) => {
    //     const files = Array.from(event.target.files);
    //     for (let file of files) {
    //         await addPdfToDatabase(file);
    //     }
    //     loadPdfs();
    // };

    const handleFileChange = async () => {
        await loadPdfs(); // Reload PDFs after file change
    };

    const handleCheckboxChange = (event, id) => {
        console.log('event: ', event.target.checked);

        let selectedPdfs = localStorage.getItem('selectedPdfs') ? JSON.parse(localStorage.getItem('selectedPdfs')) : [];
        console.log('selectedPdfs: ', selectedPdfs);
        if (event.target.checked) {
            if (selectedPdfs.indexOf(id) === -1){
                selectedPdfs.push(id);
            }  
        }
        else {
            selectedPdfs = selectedPdfs.filter(pdfId => pdfId !== id);
        }

        localStorage.setItem('selectedPdfs', JSON.stringify(selectedPdfs));

        console.log('id: ', id);
        const updatedPdfs = pdfList.map(pdf => pdf.id === id ? { ...pdf, selected: !pdf.selected } : pdf);
        console.log('updatedPdfs: ', updatedPdfs[0].selected);
        //was commented out \/
        setPdfList(updatedPdfs);

    };

    const handleRemovePdf = async (id) => {
        await deletePdfById(id);
        const updatedPdfs = pdfList.filter(pdf => pdf.id !== id);
        setPdfList(updatedPdfs);
        
        let selectedPdfs = localStorage.getItem('selectedPdfs') ? JSON.parse(localStorage.getItem('selectedPdfs')) : [];
        console.log('selectedPdfs: ', selectedPdfs);
        selectedPdfs = selectedPdfs.filter(pdfId => pdfId !== id);
        localStorage.setItem('selectedPdfs', JSON.stringify(selectedPdfs));
    };
    
    
    return (
        <>
            <Header/>
            <div className="insights-container">
                <div className={`sidebar ${isSidebarOpen ? 'expanded' : 'collapsed'}`}>
                    <button className="toggle-btn" onClick={toggleSidebar}>
                        {isSidebarOpen ? '<' : '>'}
                    </button>
                    {isSidebarOpen && (
                        <div>
                            <div className="sidebar-menu">
                                <h1>Menu</h1>
                                <div className='sidebar-menu-buttons'>
                                    {/* <button className='menu-button' onClick={() => setActiveButton('home')}>Home</button> */}
                                    <button className='menu-button' onClick={() => setActiveButton('keyword_count')}>Keyword Count</button>
                                    <button className='menu-button' onClick={() => setActiveButton('ner')}>Named Entity Recognition</button>
                                    <button className='menu-button' onClick={() => setActiveButton('topic_modeling')}>Topic Modeling</button>
                                    <button className='menu-button' onClick={() => setActiveButton('q_a')}>Q&A</button>
                                </div>
                            </div>
                                

                            <div className="sidebar-upload">
                                <Upload
                                    loading={loading}
                                    pdfList={pdfList}
                                    onFileChange={handleFileChange}
                                    onCheckboxChange={handleCheckboxChange}
                                    onPdfRemove={handleRemovePdf}
                                />
                            </div>
                        </div>
                    )}
                </div>
                {/* <Sidebar 
                    loading={loading}
                    isExpanded={isSidebarOpen}
                    toggleSidebar={toggleSidebar}
                    pdfList={pdfList}
                    onFileChange={handleFileChange}
                    onCheckboxChange={handleCheckboxChange}
                    onPdfRemove={handleRemovePdf}
                /> */}
                {/* <div className="box upload-box">
                    <Upload 
                        loading={loading}
                        onPdfRemoved={handlePdfRemoved}
                        onDeletePdf={handleDeletePdf}
                    />
                </div> */}
                {/* <div className="right-side"> */}
                <div className={`main-content ${isSidebarOpen ? 'expanded' : ''}`}>
                    {/* <div className="box options-box">
                        <Options activeButton={activeButton} setActiveButton={setActiveButton} />
                    </div> */}
                    <div className="box output-box">
                        <Output activeButton={activeButton} />
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default GetInsights;
