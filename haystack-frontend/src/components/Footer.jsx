import './Footer.css'
import IQSpatiallogo from '../assets/logo_IQSpatial.png'; 

function Footer() {
    return (
        <div className='footer-container'>
            <img src={IQSpatiallogo} alt="The orange logo of IQSpatial" className="IQSpatialogo"/>
        </div>
    )
}

export default Footer;