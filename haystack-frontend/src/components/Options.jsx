import React from 'react';
import './Options.css';

function Options({ activeButton, setActiveButton }) {
    const handleToggle = (button) => {
        setActiveButton(button);
    };

    return (
        <div className="options-buttons">
            <button 
                className={`option-button ${activeButton === 'left' ? 'active' : ''}`}
                onClick={() => handleToggle('left')}
            >
                Left
            </button>
            <button 
                className={`option-button ${activeButton === 'right' ? 'active' : ''}`}
                onClick={() => handleToggle('right')}
            >
                Right
            </button>
            <div className={`toggle-indicator ${activeButton}`}></div>
        </div>
    );
}

export default Options;
