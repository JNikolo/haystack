// src/pages/Home.jsx
import React, { useEffect } from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
import homelogo from '../assets/homelogo.png'; 
import './Home.css';

function Home() {
    useEffect(() => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'image';
        link.href = homelogo;
        document.head.appendChild(link);

        return () => {
            document.head.removeChild(link);
        };
    }, []);

    return (
        <>
            <Header />
            <div className='home-content'>
                <img src={homelogo} alt="A orange picture of a magnifying glass on a pdf." className="homelogo" />
                <h1>Instant Insights, Effortlessly.</h1>
            </div>
            <Footer />
        </>
    );
}

export default Home;

