import React from 'react';
import Header from '../components/Header';
import Footer from "../components/Footer";
import homelogo from '../assets/homelogo.png'; 
import './Home.css';

function Home({}) {
    return (
        <>
            <Header></Header>
            <div className='home-content'>
                <img src={homelogo} alt="A orange picture of a magnifying glass on a pdf." className="homelogo"/>
                <h1>Instant Insights, Effortlessly.</h1>
            </div>
            <Footer></Footer>
        </>
        
    );
}

export default Home;
