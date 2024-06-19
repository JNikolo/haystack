import React from "react";
import "./SignIn.css"; 

function SignIn() {
    const [signIn, toggle] = React.useState(true);

    return (
        <div className="container">
            <div className={`sign-up-container ${signIn ? "" : "active"}`}>
                <form className="form">
                    <h1 className="title">Create Account</h1>
                    <input type="text" placeholder="Name" className="input" />
                    <input type="email" placeholder="Email" className="input" />
                    <input type="password" placeholder="Password" className="input" />
                    <button className="button">Sign Up</button>
                </form>
            </div>

            <div className={`sign-in-container ${signIn ? "active" : ""}`}>
                <form className="form">
                    <h1 className="title">Sign in</h1>
                    <input type="email" placeholder="Email" className="input" />
                    <input type="password" placeholder="Password" className="input" />
                    <button className="button">Sign In</button>
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
                        <h1 className="title">Hello, Friend!</h1>
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
    );
}

export default SignIn;