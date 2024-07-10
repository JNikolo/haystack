// src/App.jsx
import React from 'react';
import { Navigate, BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import SignIn from './pages/SignIn';
import GetInsights from './pages/GetInsights';
import { useAuth } from './contexts/AuthContext';
import { verifyLogin } from './firebase/auth';
import './App.css';

function App() {
    

    return (
        <Router>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/signin" element={<SignIn />} />
                <Route path="/getinsights"
                    element={
                        <RequireAuth>
                            <GetInsights />
                        </RequireAuth>
                    } 
                />
            </Routes>
        </Router>
    );
}

async function RequireAuth({ children }) {
    //const { userLoggedIn } = useAuth();
  //  const loggedIn = window.localStorage.getItem("isLogged");
    loggedIn = verifyLogin();
//    return loggedIn ? children : <Navigate to="/signin" replace />;

    return loggedIn ? children : <Navigate to="/signin" replace />;
}

export default App;
