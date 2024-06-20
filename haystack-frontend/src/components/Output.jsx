import React from 'react';

function Output({ imageData }) {
    return (
        <>
            {imageData && (
                <div>
                    <h2>Generated Plot</h2>
                    <img src={`data:image/png;base64,${imageData}`} alt="Generated Plot" />
                </div>
            )}
        </>
    );
}

export default Output;
