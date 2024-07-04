import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { signOutWrapper } from '../firebase/auth';
import './Header.css'; 

function Header() {
    const isLogged = window.localStorage.getItem("isLogged");
    const { userLoggedIn } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        window.localStorage.removeItem("isLogged");
        signOutWrapper().then(() => {
            navigate("/signin"); // Redirect to sign-in page after sign out
        }).catch((error) => {
            console.error("Error signing out:", error);
        });
    }

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
                        {isLogged && (
                            <Link className='button-container' to='/getinsights'>
                                Insights
                            </Link>
                        )}
                    </p>
                </div>
                {
                    userLoggedIn
                    ? 
                    <button className='button-container' onClick={handleLogout}>Sign Out</button>
                    :
                    <p>
                        <Link className='button-container' to="/signin">Sign In</Link>
                    </p>
                }
                
            
        </div>
    );
}

export default Header;
