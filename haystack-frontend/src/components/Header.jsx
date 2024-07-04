import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css'; 

function Header() {
    return (
        <div className='header-container'>
                <div className='brand-container'>
                    <Link to="/">
                        <img src='../H-logo.png' width={40} height={40} />
                    </Link>
                    <p className='navButtons'>
                        <Link className='button-container' to='/'>
                            FAQ
                        </Link>
                        <Link className='button-container' to='/getinsights'>
                            Insights
                        </Link>
                    </p>
                </div>
                
                <p>
                    <Link className='button-container' to="/signin">Sign In</Link>
                </p>
            
        </div>
    );
}

export default Header;
