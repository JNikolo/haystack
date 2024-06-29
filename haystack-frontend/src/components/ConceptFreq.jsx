import React, { useState } from "react";
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';
import './ConceptFreq.css';

const labelDefinitions = {
    ORG: "Companies, agencies, institutions, etc.",
    PERSON: "People, including fictional.",
    GPE: 'Countries, cities, states.', 
    PRODUCT: 'Objects, vehicles, foods, etc. (Not services.)',
    EVENT: 'Named hurricanes, battles, wars, sports events, etc.'
    // Add other label definitions here
};


function ConceptFreq({ pdfList }) {
    const [isLoading, setIsLoading] = useState(false);
    const [plotData, setPlotData] = useState(null);
    const [error, setError] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);


    const handleGeneratePlots = async () => {
        setIsLoading(true);

        const formData = new FormData();
        for (let pdf of pdfList) {
            formData.append('files', pdf.file);
        }

        try {
            // Replace with your actual API endpoint for generating plots
            const response = await fetch('http://127.0.0.1:8000/conceptsfrequencies/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to generate plots');
            }

            const data = await response.json();
            setPlotData(data.results);
            setError(null); // Reset error state
        } catch (error) {
            setError(error.message || 'Failed to generate plots');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearPlot = () => {
        setPlotData(null);
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
            <h2>Concept Frequency</h2>
            <button onClick={handleGeneratePlots} disabled={isLoading || pdfList.length === 0}>
                Generate Plot
            </button>
            
            {isLoading && <p>Loading...</p>}
            {error && <p className="error-message">{error}</p>}
            {plotData && (
                <>
                    <button onClick={handleClearPlot}>Clear</button>
                    <div className="plot-container">
                        <Bar data={chartData} options={chartOptions} />
                    </div>
                </>
                
            )}
            <button onClick={handleOpenModal} className="definitions-button">Show NER Label Definitions</button>
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