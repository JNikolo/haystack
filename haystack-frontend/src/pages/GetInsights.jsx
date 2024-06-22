import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
import Upload from "../components/Upload";
import Output from "../components/Output";
import './GetInsights.css';
import axios from 'axios';
import { getPdfById, clearDatabase } from '../utils/indexDB';

function GetInsights() {
    const [imageData, setImageData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const handleBeforeUnload = (event) => {
            clearDatabase().then(() => {
                console.log("Database cleared on session end");
            });
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        // Cleanup function to remove the event listener
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

    const handleSubmit = async (selectedPdfs) => {
        setLoading(true);

        const formData = new FormData();
        for (let pdf of selectedPdfs) {
            const pdfRecord = await getPdfById(pdf.id);
            formData.append('files', pdfRecord.file);
        }

        try {
            const response = await axios.post('http://localhost:8000/conceptsfrequencies/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setImageData(response.data.image_data);
        } catch (error) {
            console.error('Error uploading files:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Header />
            <div className="insights-container">
                <div className="box upload-box">
                    <Upload 
                        loading={loading}
                        handleSubmit={handleSubmit}
                    />
                </div>
                <div className="box output-box">
                    <Output imageData={imageData} />
                </div>
            </div>
            <Footer />
        </>
    );
}

export default GetInsights;




