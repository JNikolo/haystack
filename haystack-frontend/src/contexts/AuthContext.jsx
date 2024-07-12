import React, { createContext, useState, useEffect, useContext } from 'react';
import { auth } from "../firebase/config"; // Import Firebase auth instance
import { postIdTokenToSessionLogin } from '../firebase/auth'
import { onAuthStateChanged } from "firebase/auth";

const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }){
    const [currentUser, setCurrentUser] = useState(null);
    const [userLoggedIn, setUserLoggedIn] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, initializeUser);
        return unsubscribe;
    }, []);

    async function initializeUser(user) {
        if (user) {
            //setCurrentUser({...user});
            const idToken= await user.getIdToken();
            await postIdTokenToSessionLogin('http://127.0.0.1:8000/session_login', idToken);//, csrfToken);
            //console.log("logged in: ", user)
            setUserLoggedIn(true);
        } else {
            //console.log("logged out: ",user)
            //setCurrentUser(null);
            setUserLoggedIn(false);
        }
        setLoading(false);
    }

    const value = {
        currentUser,
        userLoggedIn,
        loading
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

