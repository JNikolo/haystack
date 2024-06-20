import React, { useState } from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
import Upload from "../components/Upload";
import Output from "../components/Output";
import './GetInsights.css';
import axios from 'axios';

function GetInsights() {
    const [selectedFiles, setSelectedFiles] = useState(null);
    const [imageData, setImageData] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setSelectedFiles(event.target.files);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);

        const formData = new FormData();
        for (let i = 0; i < selectedFiles.length; i++) {
            formData.append('files', selectedFiles[i]);
        }

        try {
            //Make sure to add post address
            const response = await axios.post('', formData, {
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
                        selectedFiles={selectedFiles}
                        loading={loading}
                        handleFileChange={handleFileChange}
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


