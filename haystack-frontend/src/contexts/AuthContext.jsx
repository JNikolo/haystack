import React, { createContext, useState, useEffect, useContext } from 'react';
import { auth, analytics } from "../firebase/config"; // Import Firebase auth instance
import { postIdTokenToSessionLogin, signOutCookie } from '../firebase/auth'
import { onAuthStateChanged } from "firebase/auth";
import { logEvent } from 'firebase/analytics';

const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }){
    //const [currentUser, setCurrentUser] = useState(null);
    const [userLoggedIn, setUserLoggedIn] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, initializeUser);
        return unsubscribe;
    }, []);

    async function initializeUser(user) {
        if (user) {
            console.log("intializing user...")
            console.log("logged in: ", user)
            //setCurrentUser({...user});
            //const idToken= await user.getIdToken();
            //const idToken = await auth.currentUser.getIdToken().then(postIdTokenToSessionLogin);
            // await postIdTokenToSessionLogin(idToken);//, csrfToken);
            //console.log("logged in: ", user)
            logEvent(analytics, "login")
            setUserLoggedIn(true);
        } else {
            console.log("else statement...")
            console.log("logged out: ",user)
            //console.log("logged out: ",user)
            //setCurrentUser(null);
            //const signOutResponse = await signOutCookie();
            //console.log(signOutResponse);
            logEvent(analytics, "logout")
            setUserLoggedIn(false);
        }
        setLoading(false);
    }

    const value = {
        //currentUser,
        userLoggedIn,
        loading
    };

    return (
        <AuthContext.Provider value={value}>
            { !loading && children }
        </AuthContext.Provider>
    );
};

