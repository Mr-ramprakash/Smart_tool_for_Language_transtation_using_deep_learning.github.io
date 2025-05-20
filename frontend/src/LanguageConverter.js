import React, { useState } from "react";
import axios from "axios";

const LanguageConverter = () => {
  const [file, setFile] = useState(null);
  const [translated, setTranslated] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [fromLang, setFromLang] = useState("en");
  const [toLang, setToLang] = useState("tam_Taml");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);  // Fixed: Changed from [1] to [0] to get the first selected file
  };

  const handleTranslate = async () => {
    if (!file) {
      alert("Please select a document file first.");
      return;
    }

    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("from_lang", fromLang);
    formData.append("to_lang", toLang);

    try {
      const response = await axios.pos t("http://localhost:8000/translate_file", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response, "response");
      setTranslated(response.data.translated_text);
      setIsLoading(false);
    } catch (error) {
      console.error("Translation Error:", error);
      alert("Translation failed. Please check the console for more info.");
      setIsLoading(false);
    }
  };

  const backgroundStyle = {
    minHeight: "100vh",
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    padding: "40px",
    color: "#000",
    fontFamily: "Arial, sans-serif",
  };

  const boxStyle = {
    background: "rgba(255, 255, 255, 0.9)",
    padding: "20px",
    borderRadius: "10px",
    maxWidth: "600px",
    margin: "auto",
    boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
  };

  return (
    <div style={backgroundStyle}>
      <div style={boxStyle}>
        <h2>AI-Powered Document Translator</h2>

        {/* File Upload */}
        <input 
          type="file" 
          accept=".txt,.pdf,.doc,.docx" 
          onChange={handleFileChange} 
        />
        <br /><br />

        {/* Language Selection */}
        <label>From: </label>
        <select value={fromLang} onChange={(e) => setFromLang(e.target.value)}>
          <option value="en">English</option>
          
        </select>

        <label style={{ marginLeft: "20px" }}>To: </label>
        <select value={toLang} onChange={(e) => setToLang(e.target.value)}>
          <option value="tam_Taml">Tamil</option>
          <option value="tel_Telu">Telugu</option>
          <option value="kan_Knda">Kannada</option>
          <option value="mal_Mlym">Malayalam</option>
        </select>

        <br /><br />

        {/* Translate Button with Spinner */}
        <button
          onClick={handleTranslate}
          disabled={isLoading}
          style={{
            position: "relative",
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#2196f3",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: isLoading ? "not-allowed" : "pointer",
          }}
        >
          {isLoading ? (
            <span
              style={{
                border: "3px solid #f3f3f3",
                borderTop: "3px solid #fff",
                borderRadius: "50%",
                width: "20px",
                height: "20px",
                animation: "spin 1s linear infinite",
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
              }}
            />
          ) : (
            "Translate"
          )}
        </button>

        {/* Improved Translated Text Display */}
        <div style={{ marginTop: "20px" }}>
          <strong>Translation:</strong>
          <div
            style={{
              maxHeight: "300px",
              overflowY: "auto",
              padding: "15px",
              marginTop: "10px",
              border: "1px solid #ddd",
              borderRadius: "5px",
              backgroundColor: "#f9f9f9",
              whiteSpace: "pre-wrap",
              textAlign: "left",
              minHeight: "50px"
            }}
          >
            {translated || "No translation yet."}
          </div>
          
          {/* Character count display */}
          {translated && (
            <div style={{
              textAlign: "right",
              color: "#666",
              fontSize: "0.8em",
              marginTop: "5px"
            }}>
              {translated.length} characters
            </div>
          )}
          
          {/* Copy button */}
          {translated && (
            <button
              onClick={() => {
                navigator.clipboard.writeText(translated);
                alert("Translation copied to clipboard!");
              }}
              style={{
                marginTop: "10px",
                padding: "5px 10px",
                backgroundColor: "#4CAF50",
                color: "white",
                border: "none",
                borderRadius: "3px",
                cursor: "pointer"
              }}
            >
              Copy Translation
            </button>
          )}
        </div>
      </div>

      {/* Spinner Keyframe CSS */}
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default LanguageConverter;
