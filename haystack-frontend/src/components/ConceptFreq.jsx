import React, { useEffect, useState } from "react";
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';
import './ConceptFreq.css';
import { getPdfById } from '../utils/indexedDB';
import Loading from "./Loading";


const labelDefinitions = {
    ORG: "Companies, agencies, institutions, etc.",
    PERSON: "People, including fictional.",
    GPE: 'Countries, cities, states.', 
    PRODUCT: 'Objects, vehicles, foods, etc. (Not services.)',
    EVENT: 'Named hurricanes, battles, wars, sports events, etc.'
    // Add other label definitions here
};


function ConceptFreq({}) {
    const [isLoading, setIsLoading] = useState(false);
    const [plotData, setPlotData] = useState(null);
    const [error, setError] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    useEffect(() => {
        const plotData = JSON.parse(localStorage.getItem('plotData'));
        if (plotData) {
            setPlotData(plotData);
        }
        else{
            setPlotData(null);
        }
    }, []);


    const handleGeneratePlots = async () => {
        setIsLoading(true);
        const selectedPdfs = JSON.parse(localStorage.getItem('selectedPdfs'));

        if (!selectedPdfs || selectedPdfs.length === 0) {
            alert('Please select PDFs');
            setIsLoading(false);
            return;
        }

        // Filter selected PDFs
        //const selectedPdfs = pdfList.filter(pdf => pdf.selected);

        const formData = new FormData();
        // for (let pdf of selectedPdfs) {
        //     formData.append('files', pdf.file);
        // }

        for (let pdfID of selectedPdfs) {
            const pdf = await getPdfById(pdfID);
            if (pdf) {
                formData.append('files', pdf.file);
            }
            else{
                console.log('PDF not found in indexedDB');
                const filteredPdfs = selectedPdfs.filter(pdf => pdf !== pdfID);
                localStorage.setItem('selectedPdfs', JSON.stringify(filteredPdfs));
                setIsLoading(false);
                setError('A PDF was not found! Try again.');
                return;
            }
        }

        try {
            // Replace with your actual API endpoint for generating plots
            const response = await fetch('http://127.0.0.1:8000/conceptsfrequencies/', {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'include',
            });

            if (!response.ok) {
                throw new Error('Failed to generate plots');
            }

            const data = await response.json();
            setPlotData(data.results);
            localStorage.setItem('plotData', JSON.stringify(data.results));
            setError(null); // Reset error state
        } catch (error) {
            setError(error.message || 'Failed to generate plots');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearPlot = () => {
        setPlotData(null);
        setError(null);
        localStorage.removeItem('plotData');
    };

    // Define a color palette
    const colorPalette = [
        'rgba(75, 192, 192, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(255, 206, 86, 0.6)',
        'rgba(153, 102, 255, 0.6)',
        'rgba(255, 159, 64, 0.6)',
        'rgba(199, 199, 199, 0.6)',
        'rgba(83, 102, 255, 0.6)',
        'rgba(255, 153, 153, 0.6)',
        'rgba(75, 192, 192, 0.6)'
    ];

    const chartData = {
        labels: plotData ? plotData.map(item => item.text) : [],
        datasets: [
            {
                label: 'Frequency',
                data: plotData ? plotData.map(item => item.frequency) : [],
                backgroundColor: plotData ? plotData.map((_, index) => colorPalette[index % colorPalette.length]) : [],
                borderColor: plotData ? plotData.map((_, index) => colorPalette[index % colorPalette.length].replace('0.6', '1')) : [],
                borderWidth: 1,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        scales: {
            x: {
                beginAtZero: true,
            },
            y: {
                beginAtZero: true,
                ticks: {
                    precision: 0,
                    stepSize: 10, // Set the step size to 20 units
                },
            },
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const index = context.dataIndex;
                        const frequency = context.raw;
                        const labels = plotData[index].labels.join(', ');
                        return `Frequency: ${frequency}, NER Labels: ${labels}`;
                    }
                }
            },
        }
    };

    const handleOpenModal = () => {
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
    };



    return (
        <div className="concept-freq">
            <h2>Perform Entity Recognition on Your PDFs!</h2>
            <button className='frequency-button' onClick={handleGeneratePlots} disabled={isLoading }>
                Generate Plot
            </button>
            <button className='frequency-button' onClick={handleOpenModal}>
                Show NER Label Definitions
            </button>
            {isLoading && (
                <div className="loading-concept">
                    <Loading />
                </div>
            )}
            {error && <p className="error-message">{error}</p>}
            {plotData && (
                    <div className="plot-container">
                        <Bar data={chartData} options={chartOptions} width={800}/>
                    </div>
            )}
             
            <button className='frequency-button' onClick={handleClearPlot}>Clear</button>
            {isModalOpen && (
                <div className="modal">
                    <div className="modal-content">
                        <span className="close-button" onClick={handleCloseModal}>&times;</span>
                        <h3>Label Definitions</h3>
                        <ul>
                            {Object.entries(labelDefinitions).map(([label, definition]) => (
                                <li key={label}><strong>{label}:</strong> {definition}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ConceptFreq;