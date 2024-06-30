import React from "react";
import Upload from "./Upload";
import "./Sidebar.css";

function Sidebar({ loading, isExpanded, toggleSidebar, pdfList, onFileChange, onCheckboxChange, onPdfRemove }) {
    return (
        <div className={`sidebar ${isExpanded ? 'expanded' : 'collapsed'}`}>
            <button className="toggle-btn" onClick={toggleSidebar}>
                {isExpanded ? '<' : '>'}
            </button>
            {isExpanded && (
                <Upload
                    loading={loading}
                    pdfList={pdfList}
                    onFileChange={onFileChange}
                    onCheckboxChange={onCheckboxChange}
                    onPdfRemove={onPdfRemove}
                />
            )}
        </div>
    );
}

export default Sidebar;
