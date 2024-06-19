import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css'; 

function Header() {
    return (
        <div className='header-container'>
            <div className='brand-container'>
                <h1>
                    <Link to="/">Haystack</Link>
                </h1>
            </div>
            <div className='button-container'>
                <h1>
                    <Link to="/signin">Sign In</Link>
                </h1>
            </div>
        </div>
    );
}

export default Header;
