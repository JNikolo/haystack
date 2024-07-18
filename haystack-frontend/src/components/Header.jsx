import React, {useState, useEffect} from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { signOutWrapper, verifyLogin } from '../firebase/auth';
import './Header.css'; 

function Header() {

    //const [isLogged, setIsLogged] = useState(null);
    const navigate = useNavigate();

    // useEffect( () =>{
    //     verifyLogin()
    //         .then( (response) => {
    //             setIsLogged(response);
    //         }).catch((error) => {
    //             console.log("Error verifying login status: ", error);
    //             setIsLogged(false);
    //         })
    // } ,[]);
    

    //const isLogged = window.localStorage.getItem("isLogged");
    const { userLoggedIn } = useAuth();
    //const isLogged = verifyLogin();
    // console.log(isLogged);

    const handleLogout = () => {
        signOutWrapper().then(() => {
            //window.localStorage.removeItem("isLogged");
            navigate("/signin"); // Redirect to sign-in page after sign out
        }).catch((error) => {
            console.error("Error signing out:", error);
        });
    }

    // if (isLogged === null) {
    //     return (<div>Loading...</div>); //or loading indicator
    // }

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
                        {userLoggedIn && ( //isLogged && (
                            <Link className='button-container' to='/getinsights'>
                                Insights
                            </Link>
                        )}
                    </p>
                </div>
                {
                    userLoggedIn
                    // isLogged
                    ? 
                    <p>
                        <button className='sign-out-button' onClick={handleLogout}>Sign Out</button>
                    </p>
                    :
                    <p>
                        <Link className='button-container' to="/signin">Sign In</Link>
                    </p>
                }
                
            
        </div>
    );
}

export default Header;
