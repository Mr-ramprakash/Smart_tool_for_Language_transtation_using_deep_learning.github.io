import React, { useState, useEffect } from "react";
import axios from "axios";

const LanguageConverter = () => {
  const [file, setFile] = useState(null);
  const [translated, setTranslated] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [fromLang, setFromLang] = useState("en");
  const [toLang, setToLang] = useState("ta");
  const [domain, setDomain] = useState("");
  const [preserveLayout, setPreserveLayout] = useState(false);
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const [domains, setDomains] = useState(["General", "Legal", "Medical", "Technical"]);

  useEffect(() => {
    // Fetch supported languages
    axios.get("http://localhost:8000/supported_languages")
      .then(response => {
        setSupportedLanguages(response.data.languages);
      })
      .catch(error => {
        console.error("Error fetching languages:", error);
      });
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
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
    formData.append("domain", domain);
    formData.append("preserve_layout", preserveLayout);

    try {
      const response = await axios.post(
        "http://localhost:8000/translate_document", 
        formData, 
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      
      setTranslated(response.data.translated_text);
      setIsLoading(false);
      
      // Handle layout information if returned
      if (response.data.layout_info) {
        console.log("Layout information:", response.data.layout_info);
        // Here you would implement logic to reconstruct the document with layout
      }
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
    maxWidth: "800px",
    margin: "auto",
    boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
  };

  return (
    <div style={backgroundStyle}>
      <div style={boxStyle}>
        <h2>Enhanced AI-Powered Document Translator</h2>
        
        {/* File Upload */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "5px" }}>
            Document to Translate:
          </label>
          <input 
            type="file" 
            accept=".txt,.doc,.docx" 
            onChange={handleFileChange}
            style={{ width: "100%" }}
          />
        </div>

        {/* Language Selection */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginBottom: "15px" }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: "5px" }}>From: </label>
            <select 
              value={fromLang} 
              onChange={(e) => setFromLang(e.target.value)}
              style={{ width: "100%", padding: "8px" }}
            >
              {supportedLanguages.filter(lang => lang.code !== "ta").map(lang => (
                <option key={lang.code} value={lang.code}>{lang.name}</option>
              ))}
            </select>
          </div>
          
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: "5px" }}>To: </label>
            <select 
              value={toLang} 
              onChange={(e) => setToLang(e.target.value)}
              style={{ width: "100%", padding: "8px" }}
            >
              {supportedLanguages.filter(lang => lang.code !== "en").map(lang => (
                <option key={lang.code} value={lang.code}>{lang.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Domain Selection */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "5px" }}>Domain (optional): </label>
          <select 
            value={domain} 
            onChange={(e) => setDomain(e.target.value)}
            style={{ width: "100%", padding: "8px" }}
          >
            <option value="">General</option>
            {domains.map(domain => (
              <option key={domain} value={domain.toLowerCase()}>{domain}</option>
            ))}
          </select>
        </div>

        {/* Layout Preservation */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "flex", alignItems: "center" }}>
            <input 
              type="checkbox" 
              checked={preserveLayout}
              onChange={(e) => setPreserveLayout(e.target.checked)}
              style={{ marginRight: "10px" }}
            />
            Preserve Document Layout (tables, formatting)
          </label>
        </div>

        {/* Translate Button */}
        <button
          onClick={handleTranslate}
          disabled={isLoading}
          style={{
            width: "100%",
            padding: "12px",
            fontSize: "16px",
            backgroundColor: "#2196f3",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: isLoading ? "not-allowed" : "pointer",
            position: "relative",
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
                display: "inline-block",
              }}
            />
          ) : (
            "Translate Document"
          )}
        </button>

        {/* Translated Text Display */}
        <div style={{ marginTop: "20px" }}>
          <strong>Translation Result:</strong>
          <div style={{ 
            marginTop: "10px",
            padding: "15px",
            border: "1px solid #ddd",
            borderRadius: "5px",
            minHeight: "100px",
            background: "#f9f9f9",
            whiteSpace: "pre-wrap"
          }}>
            {translated || "No translation yet. Upload a document and click Translate."}
          </div>
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