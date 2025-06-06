// src/pages/SignIn.jsx
import React, { useState, useEffect } from "react";
//import { auth } from "../firebase/config";
import { useAuth } from "../contexts/AuthContext";
import { signInWrapper, createUserWrapper } from "../firebase/auth";
import { useNavigate, Navigate } from "react-router-dom";
import { verifyLogin } from '../firebase/auth'
import "./SignIn.css";
import Header from '../components/Header';
import { BiSolidError } from "react-icons/bi";

function SignIn() {
    const { userLoggedIn } = useAuth();
   // const isLogged = window.localStorage.getItem("isLogged");
    //const isLogged = verifyLogin();
    const [signIn, toggle] = useState(true);
    const [email, setEmail] = useState("");
    const [emailError, setEmailError] = useState(false);
    const [credentialError, setCredentialError] = useState(false);
    const [password, setPassword] = useState("");
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

    const validateEmail = (email) => {
        const re = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
        if (!re.test(String(email).toLowerCase())) {
          return 'Invalid email format.';
        }
        return null;
    }

    const handleSignIn = (e) => {
        
        e.preventDefault();
        const emailError = validateEmail(email);
        if (emailError) {
            console.error("Invalid email format.");
            setEmailError(true);
            return;
        }
        else {
            setEmailError(false);
        }
        
        signInWrapper(email, password)
        .then(() => {
            setCredentialError(false);
            navigate("/"); // Redirect to GetInsights page on successful sign-in
        })
        .catch(error => {
            console.log('Error signing in:', error);
            if (error.code==="auth/invalid-credential"){
                console.log("Invalid credentials");
                setCredentialError(true);
                return;
            }
            // Handle the error appropriately, e.g., display an error message to the user
        });
        
        
        
        // signInWithEmailAndPassword(auth, email, password)
        //     .then((userCredential) => {
        //         // Signed in
        //         const user = userCredential.user;
        //         console.log("Signed in as:", user.email);

        //         navigate("/getinsights"); // Redirect to GetInsights page
        //     })
        //     .catch((error) => {
        //         console.error("Error signing in:", error);
        //     });
    };

    const handleSignUp = (e) => {
        e.preventDefault();
        createUserWrapper(email, password);
        // createUserWithEmailAndPassword(auth, email, password)
        //     .then((userCredential) => {
        //         // Signed up
        //         const user = userCredential.user;
        //         console.log("Signed up as:", user.email);
        //     })
        //     .catch((error) => {
        //         console.error("Error signing up:", error);
        //     });
    };

    // if (isLogged){
    //     return (<Navigate to="/getinsights" />);
    // }

    if (userLoggedIn){
        return (<Navigate to="/" />);
    }

    const handleInputChange = (e) => {
        setEmailError(false);
        setCredentialError(false);
        if (e.target.type === 'email') setEmail(e.target.value);
        if (e.target.type === 'password') setPassword(e.target.value);
    };

    return (
        <>
        {/*isLogged && (<Navigate to="/getinsights" />)*/}
        {/*(<Navigate to="/getinsights" />)*/}
        <div className="container">
            <Header></Header>
            <div className={`sign-up-container ${signIn ? "" : "active"}`}>
                <form className="form" onSubmit={handleSignUp}>
                    <h1 className="title">Create Account</h1>
                    <input type="text" placeholder="Name" className="input" />
                    <input type="email" placeholder="Email" className="input" onChange={(e) => setEmail(e.target.value)} />
                    <input type="password" placeholder="Password" className="input" onChange={(e) => setPassword(e.target.value)} />
                    {/* Button is disabled to prevent creating new accounts */}
                    <button className="button" type="submit" disabled={false}>Sign Up</button>
                </form>
            </div>

            <div className={`sign-in-container ${signIn ? "active" : ""}`}>
                <form className="form" onSubmit={handleSignIn}>
                    <h1 className="title">Sign in</h1>
                    {credentialError && <p className="signin-error"><BiSolidError/> Invalid Credentials. Please try again!</p>}
                    {emailError && <p className="signin-error"><BiSolidError/> Invalid email format.</p>}
                    <input type="email" placeholder="Email" className="input" onChange={(e) => handleInputChange(e)} />
                    <input type="password" placeholder="Password" className="input" onChange={(e) => handleInputChange(e)} />
                    <button className="button" type="submit">Sign In</button>
                </form>
            </div>

            <div className={`overlay-container ${signIn ? "" : "active"}`}>
                <div className="overlay">
                    <div className={`overlay-panel left-overlay-panel ${signIn ? "active" : ""}`}>
                        <h1 className="title">Welcome Back!</h1>
                        <p className="paragraph">
                            Glad to see you again! Please sign in to continue your journey.
                        </p>
                        <button className="ghost-button" onClick={() => toggle(true)}>
                            Sign In
                        </button>
                    </div>
                    <div className={`overlay-panel right-overlay-panel ${signIn ? "" : "active"}`}>
                        <h1 className="title">Hello!</h1>
                        <p className="paragraph">
                            New to our site? Sign up to start your journey.
                        </p>
                        <button className="ghost-button" onClick={() => toggle(false)}>
                            Sign Up
                        </button>
                    </div>
                </div>
            </div>
        </div>
        </>
    );
}

export default SignIn;
