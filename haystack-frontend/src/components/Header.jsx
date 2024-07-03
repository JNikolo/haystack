import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css'; 

function Header() {
    return (
        <div className='header-container'>
                <div className='brand-container'>
                    <Link to="/">
                        <img src='../H-logo.png' width={60} height={60} />
                    </Link>
                </div>
                <p className='button-container'>
                    <Link to="/signin">Sign In</Link>
                </p>
        </div>
    );
}

export default Header;
