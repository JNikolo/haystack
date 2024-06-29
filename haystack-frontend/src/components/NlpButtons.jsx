import React from 'react';
import './NlpButtons.css';

function NlpButtons({ activeButtonNlp, buttonClicked, onButtonClick }) {
    return (
        <div className='nlpButtons'>
            <button
                className={`nlpButton ${activeButtonNlp === 'left' && buttonClicked ? 'active' : ''}`}
                onClick={() => onButtonClick('left')}
            >
                Keyword Counting
            </button>
            <button
                className={`nlpButton ${activeButtonNlp === 'middle' && buttonClicked ? 'active' : ''}`}
                onClick={() => onButtonClick('middle')}
            >
                Concept Frequency
            </button>
            <button
                className={`nlpButton ${activeButtonNlp === 'right' && buttonClicked ? 'active' : ''}`}
                onClick={() => onButtonClick('right')}
            >
                Topic Modeling
            </button>
        </div>
    );
}

export default NlpButtons;
