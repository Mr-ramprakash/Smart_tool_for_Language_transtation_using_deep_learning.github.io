import React, { useState } from "react";
import axios from "axios";

const LanguageConverter = () => {
  const [file, setFile] = useState(null);
  const [translated, setTranslated] = useState("");
  const [fromLang, setFromLang] = useState("en");
  const [toLang, setToLang] = useState("fr");

  // ✅ Handle file selection
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // ✅ Translate file using FastAPI backend
  const handleTranslate = async () => {
    if (!file) {
      alert("Please select a document file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("from_lang", fromLang);
    formData.append("to_lang", toLang);

    try {
      const response = await axios.post("http://localhost:8000/tamil", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response, "response");
      setTranslated(response.data.translated_text);
    } catch (error) {
      console.error("Translation Error:", error);
      alert("Translation failed. Please check the console for more info.");
    }
  };

  // ✅ Styles
  const backgroundStyle = {
    minHeight: "100vh",
    background: "linear-gradient(to bottom right, #e0f7fa, #2196f3)",
    padding: "40px",
    color: "#000",
    fontFamily: "Arial, sans-serif",
  };

  const boxStyle = {
    background: "white",
    padding: "20px",
    borderRadius: "10px",
    maxWidth: "600px",
    margin: "auto",
    boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.1)",
  };

  return (
    <div style={backgroundStyle}>
      <div style={boxStyle}>
        <h2>AI-Powered Document Translator</h2>

        {/* ✅ File Upload */}
        <input type="file" accept=".txt,.pdf,.doc,.docx" onChange={handleFileChange} />
        <br /><br />

        {/* ✅ Language Selection */}
        <label>From: </label>
        <select value={fromLang} onChange={(e) => setFromLang(e.target.value)}>
          <option value="en">English</option>
          <option value="fr">French</option>
          <option value="es">Spanish</option>
        </select>

        <label style={{ marginLeft: "20px" }}>To: </label>
        <select value={toLang} onChange={(e) => setToLang(e.target.value)}>
          <option value="ta">Tamil</option>
        </select>

        <br /><br />

        {/* ✅ Translate Button */}
        <button onClick={handleTranslate}>Translate</button>

        {/* ✅ Translated Text Display */}
        <div style={{ marginTop: "20px" }}>
          <strong>Translation:</strong>
          <p>{translated || "No translation yet."}</p>
        </div>
      </div>
    </div>
  );
};

export default LanguageConverter;
