import React from 'react';
import './Output.css';

function Output({ activeButton, imageData }) {
    return (
        <div className="output-container">
            {activeButton === 'left' && imageData && (
                <>
                    <h2>Generated Plot</h2>
                    <img src={`data:image/png;base64,${imageData}`} alt="Generated Plot" />
                </>
            )}
            {activeButton === 'right' && (
                <h1>Hello</h1>
            )}
        </div>
    );
}

export default Output;
