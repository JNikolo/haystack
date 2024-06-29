import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
import Upload from "../components/Upload";
import Output from "../components/Output";
import Options from "../components/Options";
import './GetInsights.css';
import { getPdfById, clearDatabase } from '../utils/indexedDB';

function GetInsights() {
    const [loading, setLoading] = useState(false);
    const [activeButton, setActiveButton] = useState('left');
    const [pdfList, setPdfList] = useState([]);


    useEffect(() => {
        const handleBeforeUnload = (event) => {
            clearDatabase().then(() => {
                console.log("Database cleared on session end");
            });
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

    const handleSubmit = async (selectedPdfs) => {
        setLoading(true);

        // const formData = new FormData();
      
        let newPdfList = [];
        for (let pdf of selectedPdfs) {
            const pdfRecord = await getPdfById(pdf.id);
            newPdfList.push(pdfRecord);
        }

        setPdfList(newPdfList);

        // call the API to populate vector db

        setLoading(false);
            
        //     formData.append('files', pdfRecord.file);
        //     console.log('pdf: ',formData);
        //console.log(pdfList);
    };

    const handleReset = async () => {
        setLoading(true);
        // await clearDatabase();
        setPdfList([]);

        // call API here to delete all vector DB records
        
        setLoading(false);
    };
    
    const handlePdfRemoved = async (updatedPdfs) => {
        const selectedPdfs = updatedPdfs.filter(pdf => pdf.selected);
        setPdfList(selectedPdfs);

        // call API here to delete specific vector DB record
    };

    const handleDeletePdf = async (id) => {
        await deletePdfById(id);
        const updatedPdfs = pdfList.filter(pdf => pdf.id !== id);
        setPdfList(updatedPdfs);
    };


    return (
        <>
            <Header />
            <div className="insights-container">
                <div className="box upload-box">
                    <Upload 
                        loading={loading}
                        onPdfRemoved={handlePdfRemoved}
                        onDeletePdf={handleDeletePdf}
                    />
                </div>
                <div className="right-side">
                    <div className="box options-box">
                        <Options activeButton={activeButton} setActiveButton={setActiveButton} />
                    </div>
                    <div className="box output-box">
                        <Output activeButton={activeButton} pdfList={pdfList} />
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default GetInsights;
