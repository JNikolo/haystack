import React, {useEffect, useState} from 'react';
import { getPdfById } from '../utils/indexedDB';
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';
import Loading from './Loading';
import './TopicModeling.css';

function CountChart({ data, topicId }) {
    // Filter data for the specific topic
    const topicData = data.filter(item => item.topic_id === topicId);

    // Prepare labels and word count values
    const labels = topicData.map(item => item.word);
    const wordCountValues = topicData.map(item => item.word_count);

    const chartData = {
        labels: labels,
        datasets: [
            {
                label: `Topic ${topicId} Word Count`,
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                hoverBackgroundColor: 'rgba(255, 99, 132, 0.8)',
                hoverBorderColor: 'rgba(255, 99, 132, 1)',
                data: wordCountValues,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
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
    };

    return <Bar data={chartData} options={chartOptions} />;
}

function ImportanceChart({ importance_data, topicId }) {
    const topicData = importance_data.filter(item => item.topic_id === topicId);
    // Prepare labels and importance values
    const labels = topicData.map(item => item.word);
    const importanceValues = topicData.map(item => item.importance);

    const chartData = {
        labels: labels,
        datasets: [
            {
                label: `Topic ${topicId} Importance`,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                hoverBackgroundColor: 'rgba(54, 162, 235, 0.8)',
                hoverBorderColor: 'rgba(54, 162, 235, 1)',
                data: importanceValues,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                beginAtZero: true,
            },
            y: {
                beginAtZero: true,
                ticks: {
                    precision: 2, // Increase precision to 2 decimal places
                },
            },
        },
    };

    return <Bar data={chartData} options={chartOptions} />;
}


function TopicModelingPlots({ data }) {
    const topicIds = [1, 2, 3, 4, 5];

    return (
        <div className='topic-modeling-plots'>
            <h3>Topic Modeling Plots</h3>
            {topicIds.map(topicId => (
                <div key={topicId}>
                    <h4>Topic {topicId}</h4>
                    <div className="plot-container">
                        <CountChart data={data.count} topicId={topicId} />
                    </div>
                    <div className="plot-container">
                        <ImportanceChart importance_data={data.importance} topicId={topicId} />
                    </div>
                </div>
            ))}
        </div>
    );
}

function TopicModeling({ selectedPdfs }) {
    const [isLoading, setIsLoading] = useState(false);
    const [apiData, setApiData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const topicData = JSON.parse(localStorage.getItem('topicData'));
        if (topicData) {
            setApiData(topicData);
        } else {
            setApiData(null);
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
            } else {
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
            const response = await fetch('http://127.0.0.1:8000/topicmodeling/', {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'include',
            });

            if (!response.ok) {
                throw new Error('Failed to generate plots');
            }

            const data = await response.json();
            setApiData(data.result);
            localStorage.setItem('topicData', JSON.stringify(data.result));
            setError(null); // Reset error state
        } catch (error) {
            setError(error.message || 'Failed to generate plots');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearPlot = () => {
        setApiData(null); // Reset apiData on clear
        setError(null);
        localStorage.removeItem('topicData');
    };

    return (
        <div className='topic-modeling'>
            <h2>Analyze Your PDFs with Topic Modeling!</h2>
            <button className='topic-button' onClick={handleGeneratePlots} disabled={isLoading}>
                Generate Plot
            </button>
            <button className='topic-button' onClick={handleClearPlot}>Clear</button>
            {isLoading && <Loading />}
            {error && <p className="error-message">{error}</p>}
            {apiData && apiData.count && apiData.importance ? (
                <div className='topic-modeling-plots'>
                    <TopicModelingPlots data={apiData} />
                </div>
            ) : null}
        </div>
    );
}

export default TopicModeling;
