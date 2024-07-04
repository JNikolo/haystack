// src/pages/SignIn.jsx
import React, { useState } from "react";
import { auth } from "../firebase/config";
import { useAuth } from "../contexts/AuthContext";
import { signInWrapper, createUserWrapper } from "../firebase/auth";
import { useNavigate, Navigate } from "react-router-dom";
import "./SignIn.css";
import Header from '../components/Header';

function SignIn() {
    const { userLoggedIn } = useAuth();
    const [signIn, toggle] = useState(true);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    const handleSignIn = (e) => {
        window.localStorage.setItem("isLogged", true);
        e.preventDefault();
        signInWrapper(email, password).then(() => {
            navigate("/getinsights"); // Redirect to GetInsights page
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

    return (
        <>
        {userLoggedIn && (<Navigate to="/getinsights" />)}
        <div className="container">
            <Header></Header>
            <div className={`sign-up-container ${signIn ? "" : "active"}`}>
                <form className="form" onSubmit={handleSignUp}>
                    <h1 className="title">Create Account</h1>
                    <input type="text" placeholder="Name" className="input" />
                    <input type="email" placeholder="Email" className="input" onChange={(e) => setEmail(e.target.value)} />
                    <input type="password" placeholder="Password" className="input" onChange={(e) => setPassword(e.target.value)} />
                    {/* Button is disabled to prevent creating new accounts */}
                    <button className="button" type="submit" disabled={true}>Sign Up</button>
                </form>
            </div>

            <div className={`sign-in-container ${signIn ? "active" : ""}`}>
                <form className="form" onSubmit={handleSignIn}>
                    <h1 className="title">Sign in</h1>
                    <input type="email" placeholder="Email" className="input" onChange={(e) => setEmail(e.target.value)} />
                    <input type="password" placeholder="Password" className="input" onChange={(e) => setPassword(e.target.value)} />
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
