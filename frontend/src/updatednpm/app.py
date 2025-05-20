from fastapi import FastAPI, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, MBart50TokenizerFast, MBartForConditionalGeneration
import nltk
from docx import Document
from typing import Annotated, Dict
import torch
import os
import io
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import easyocr
from PyPDF2 import PdfReader
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Initialize EasyOCR reader once (cache it)
reader = easyocr.Reader(['en'])  # Default English, add more languages if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt_tab')

# Initialize NLTK resources
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt/PY3/english.pickle')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')  # Additional resource for some languages

initialize_nltk()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
MODEL_CACHE: Dict[str, tuple] = {}

def load_model(from_lang: str, to_lang: str):
    """Load and cache translation model with special handling"""
    model_key = f"{from_lang}-{to_lang}"

    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    try:
        logger.info("loading")  
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Falling back to multilingual model for {model_key}: {str(e)}")
        model_name = "Helsinki-NLP/opus-mt-en-mul"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')
        logger.info(f"Model {model_key} loaded on GPU")
    else:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info(f"Model {model_key} quantized for CPU")

    MODEL_CACHE[model_key] = (model, tokenizer)
    return model, tokenizer

def process_docx_file(content: bytes) -> str:
    """Process docx file content in memory"""
    doc = Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs)

def read_pdf_pypdf2(content: bytes) -> str:
    """Read PDF using PyPDF2"""
    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_pdf_pdfplumber(content: bytes) -> str:
    """Read PDF using pdfplumber"""
    text = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def read_pdf_pdfminer(content: bytes) -> str:
    """Read PDF using pdfminer.six"""
    return extract_text(io.BytesIO(content))

def read_pdf_fitz(content: bytes) -> str:
    """Read PDF using PyMuPDF (fitz)"""
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

async def read_pdf_with_ocr(content: bytes) -> str:
    """Read PDF with OCR (for scanned PDFs)"""
    text = ""
    images = convert_from_bytes(content)
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        text += page_text + "\n"
    return text

async def translate_text_batch(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    to_lang: str,
    batch_size: int = 8
) -> list[str]:
    """Translate a batch of texts efficiently with model-specific handling"""
    translations = []
    
    def process_batch(batch):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        generate_kwargs = {
            'max_length': 128,
            'num_beams': 4,
            'early_stopping': True
        }
        
        if hasattr(tokenizer, 'convert_tokens_to_ids') and to_lang in tokenizer.get_vocab():
            generate_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(to_lang)
        elif isinstance(model, MarianMTModel):
            pass
        elif hasattr(model.config, 'decoder_start_token_id'):
            generate_kwargs['decoder_start_token_id'] = tokenizer.lang_code_to_id[to_lang]
        
        with torch.no_grad():
            translated = model.generate(**inputs, **generate_kwargs)
        
        return tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                translated = await loop.run_in_executor(executor, process_batch, batch)
                translations.extend(translated)
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                for text in batch:
                    try:
                        single_result = await loop.run_in_executor(
                            executor, process_batch, [text]
                        )
                        translations.extend(single_result)
                    except Exception as single_e:
                        logger.error(f"Failed to translate: {text[:50]}...")
                        translations.append(f"[TRANSLATION ERROR: {str(single_e)}]")
    
    return translations

@app.post("/translate_file")
async def translate_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Upload a .docx, .txt, image, or PDF file")],
    from_lang: Annotated[str, Form(description="Source language code (e.g., 'en')")],
    to_lang: Annotated[str, Form(description="Target language code (e.g., 'ta')")],
    batch_size: int = Form(8, description="Translation batch size (4-16)")
):
    """Optimized file translation endpoint"""
    try:
        content = await file.read()
        
        if file.filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: reader.readtext(content)
            text = "\n".join([result[1] for result in results])
        elif file.filename.endswith(".docx"):
            text = await asyncio.get_event_loop().run_in_executor(
                None, process_docx_file, content)
        elif file.filename.endswith(".txt"):
            text = content.decode('utf-8')
        elif file.filename.endswith(".pdf"):
            # Try different PDF readers in order of preference
            try:
                text = read_pdf_pdfplumber(content)
            except Exception as e1:
                logger.warning(f"pdfplumber failed, trying PyMuPDF: {str(e1)}")
                try:
                    text = read_pdf_fitz(content)
                except Exception as e2:
                    logger.warning(f"PyMuPDF failed, trying PyPDF2: {str(e2)}")
                    try:
                        text = read_pdf_pypdf2(content)
                    except Exception as e3:
                        logger.warning(f"PyPDF2 failed, trying OCR: {str(e3)}")
                        text = await read_pdf_with_ocr(content)
        else:
            return {"error": "Unsupported file type"}
        
        model, tokenizer = load_model(from_lang, to_lang)
        sentences = nltk.sent_tokenize(text)
        translated = await translate_text_batch(
            sentences,
            model,
            tokenizer,
            to_lang=to_lang,
            batch_size=min(max(batch_size, 4), 16)
        )
        
        return {
            "from": from_lang,
            "to": to_lang,
            "translated_text": " ".join(translated),
            "sentence_count": len(sentences)
        }
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return {"error": str(e)}

@app.get("/translate_text")
async def translate_text(
    text: str = Query(..., description="Text to be translated"),
    from_lang: str = Query("en", description="Source language"),
    to_lang: str = Query("fr", description="Target language"),
):
    """Optimized text translation endpoint"""
    try:
        model, tokenizer = load_model(from_lang, to_lang)
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) == 1:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = model.generate(**inputs)
            
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            translated = await translate_text_batch(sentences, model, tokenizer)
            result = " ".join(translated)
        
        return {"translated_text": result}
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return {"error": str(e)}
