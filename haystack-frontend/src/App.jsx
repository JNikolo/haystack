// src/App.jsx
import React, {useState, useEffect} from 'react';
import { Navigate, BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import SignIn from './pages/SignIn';
import GetInsights from './pages/GetInsights';
import { useAuth } from './contexts/AuthContext';
import { verifyLogin } from './firebase/auth';
import { clearDatabase, getAllPdfs, deletePdfById, addPdfToDatabase } from './utils/indexedDB';
import './App.css';

function App() {


    return (
        <Router>
            <Routes>
                {/* <Route path="/" element={<Home />} /> */}
                <Route path="/signin" element={<SignIn />} />
                <Route path="/"
                    element={
                        <RequireAuth>
                            <GetInsights />
                        </RequireAuth>
                    } 
                />
                <Route path="*" element={<Navigate to="/" />} />
            </Routes>
        </Router>
    );
}

function RequireAuth({ children }) {
   // const [loggedIn, setLoggedIn] = useState(null);
    const { userLoggedIn } = useAuth();
  //  const loggedIn = window.localStorage.getItem("isLogged");
//    const loggedIn = verifyLogin();
//    return loggedIn ? children : <Navigate to="/signin" replace />;
/*
    useEffect( () =>{
        verifyLogin()
            .then( (response) => {
                setLoggedIn(response);
            }).catch((error) => {
                console.log("Error verifying login status: ", error);
                setLoggedIn(false);
            })       
    } ,[]);

    if (loggedIn === null){
        return (<div>Loading...</div>); //or loading indicator
    } 
*/
    return userLoggedIn ? children : <Navigate to="/signin" replace />;
}

export default App;
