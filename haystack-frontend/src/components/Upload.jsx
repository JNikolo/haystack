import React from 'react';

function Upload({ selectedFiles, loading, handleFileChange, handleSubmit }) {
    return (
        <div className="App">
            <h1>Upload PDFs and Generate Plot</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" multiple onChange={handleFileChange} />
                <button type="submit" disabled={!selectedFiles || loading}>
                    {loading ? 'Uploading...' : 'Upload'}
                </button>
            </form>
        </div>
    );
}

export default Upload;
