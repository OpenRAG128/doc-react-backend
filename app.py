import logging
from flask import Flask, request, session, url_for, redirect, jsonify, send_file
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
import io
from bs4 import BeautifulSoup
import json
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
import gc
import tempfile
from urllib.parse import urlparse
from functools import lru_cache
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.common.exceptions import WebDriverException
import zipfile
import rarfile
import patoolib  # For RAR support 
import shutil 
import re
from werkzeug.utils import secure_filename
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import uuid
import time

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv('FLASK_SECRET_KEY', "6fK9P6WcfpBz7bWJ9qV2eP2Qv5dA8D8z")

app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

@app.before_request
def ensure_session_id():
    """Assign a unique ID to every new visitor to track their specific data."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Configuration constants
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_VIDEOS = 6
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'zip', 'rar'}  

pdf_file_path = ""


class MindmapConnector(Flowable):
    """Custom flowable for drawing connection lines between mindmap nodes"""
    def __init__(self, x1, y1, x2, y2, color):
        Flowable.__init__(self)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(1)
        self.canv.line(self.x1, self.y1, self.x2, self.y2)

def create_mindmap_pdf(markdown_content, output_path):
    """Convert markdown mindmap to PDF with interactive visualization"""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Initialize story for content
    story = []
    
    # Create base styles
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names
    custom_styles = {
        'MindmapH1': ParagraphStyle(
            'MindmapH1',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2196f3'),
            leftIndent=0
        ),
        'MindmapH2': ParagraphStyle(
            'MindmapH2',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#4caf50'),
            leftIndent=30
        ),
        'MindmapH3': ParagraphStyle(
            'MindmapH3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.HexColor('#ff9800'),
            leftIndent=60
        ),
        'MindmapBullet': ParagraphStyle(
            'MindmapBullet',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.HexColor('#f44336'),
            leftIndent=90,
            bulletIndent=75,
            firstLineIndent=0
        )
    }

    def get_level(line):
        """Determine the heading level or if it's a bullet point"""
        if line.startswith('# '):
            return 1
        elif line.startswith('## '):
            return 2
        elif line.startswith('### '):
            return 3
        elif line.startswith('- '):
            return 4
        return 0

    def clean_text(line):
        """Remove markdown symbols and clean the text"""
        return re.sub(r'^[#\- ]+', '', line).strip()

    # Process markdown content
    current_level = 0
    lines = markdown_content.split('\n')
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        level = get_level(line)
        if level == 0:
            continue
            
        text = clean_text(line)
        
        # Select style based on level
        if level == 1:
            style = custom_styles['MindmapH1']
        elif level == 2:
            style = custom_styles['MindmapH2']
        elif level == 3:
            style = custom_styles['MindmapH3']
        else:  # Bullet points
            style = custom_styles['MindmapBullet']
            text = '• ' + text
        
        # Add paragraph with appropriate style
        para = Paragraph(text, style)
        story.append(para)
        
        # Add appropriate spacing
        if level == 1:
            story.append(Spacer(1, 20))
        elif level == 2:
            story.append(Spacer(1, 15))
        else:
            story.append(Spacer(1, 10))
        
        # Add connector lines between levels
        if i > 0 and level > 1:
            story.append(MindmapConnector(
                30 * (level - 1), -15,  # Starting point
                30 * level, -5,         # Ending point
                colors.HexColor('#90caf9')  # Light blue connector
            ))
    
    # Build the PDF
    doc.build(story)
    return output_path

def create_mindmap_markdown(text):
    """Generate mindmap markdown using Gemini AI."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """
        Create a hierarchical markdown mindmap from the following text. 
        Use proper markdown heading syntax (# for main topics, ## for subtopics, ### for details).
        Focus on the main concepts and their relationships.
        Include relevant details and connections between ideas.
        Keep the structure clean and organized.
        
        Format the output exactly like this example:
        # Main Topic
        ## Subtopic 1
        ### Detail 1
        - Key point 1
        - Key point 2
        ### Detail 2
        ## Subtopic 2
        ### Detail 3
        ### Detail 4
        
        Text to analyze: {text}
        
        Respond only with the markdown mindmap, no additional text.
        """
        
        response = model.generate_content(prompt.format(text=text))
            
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error generating mindmap: {str(e)}")
        return None
    
def create_markmap_html(markdown_content):
    """Create HTML with Markmap visualization."""
    markdown_content = markdown_content.replace('`', '\\`').replace('${', '\\${')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            #mindmap {{
                width: 100%;
                height: 600px;
                margin: 0;
                padding: 0;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@6"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"></script>
    </head>
    <body>
        <svg id="mindmap"></svg>
        <script>
            window.onload = async () => {{
                try {{
                    const markdown = `{markdown_content}`;
                    const transformer = new markmap.Transformer();
                    const {{root}} = transformer.transform(markdown);
                    const mm = new markmap.Markmap(document.querySelector('#mindmap'), {{
                        maxWidth: 300,
                        color: (node) => {{
                            const level = node.depth;
                            return ['#2196f3', '#4caf50', '#ff9800', '#f44336'][level % 4];
                        }},
                        paddingX: 16,
                        autoFit: true,
                        initialExpandLevel: 2,
                        duration: 500,
                    }});
                    mm.setData(root);
                    mm.fit();
                }} catch (error) {{
                    console.error('Error rendering mindmap:', error);
                    document.body.innerHTML = '<p style="color: red;">Error rendering mindmap. Please check the console for details.</p>';
                }}
            }};
        </script>
    </body>
    </html>
    """
    return html_content



def process_compressed_file(file_path, temp_dir):
    """Extract and process files from ZIP or RAR archives"""
    extracted_text = ""
    
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        elif file_path.endswith('.rar'):
            patoolib.extract_archive(file_path, outdir=temp_dir)
        
        # Process all files in the temp directory
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        extracted_text += get_pdf_text(f) + "\n"
                elif filename.endswith('.docx'):
                    extracted_text += get_docx_text(file_path) + "\n"
                
    except Exception as e:
        logging.error(f"Error processing compressed file: {e}")
    
    return extracted_text

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def download_file(url):
    """Download file from URL with enhanced error handling and academic paper support"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,*/*',
    }
    
    try:
        if 'arxiv.org' in url:
            url = url.replace('abs', 'pdf')
            if not url.endswith('.pdf'):
                url = url + '.pdf'
        
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                return 'pdf', response.content
            elif 'docx' in content_type or url.lower().endswith('.docx'):
                return 'docx', response.content
            elif 'zip' in content_type or url.lower().endswith('.zip'):
                return 'zip', response.content
            elif 'rar' in content_type or url.lower().endswith('.rar'):
                return 'rar', response.content
            
            if response.content.startswith(b'%PDF-'):
                return 'pdf', response.content
                
        logging.error(f"Download failed for {url}. Status: {response.status_code}")
        return None, None
        
    except Exception as e:
        logging.error(f"Error downloading file from {url}: {str(e)}")
        return None, None

def process_url_file(url):
    """Enhanced URL file processing with multiple fallback methods"""
    try:
        logging.info(f"Starting to process URL: {url}")
        text = ""
        
        # Method 1: Try WebBaseLoader first (fastest)
        try:
            logging.info(f"Attempting WebBaseLoader for {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            text = "\n".join(doc.page_content for doc in docs)
            if text.strip():
                logging.info(f"WebBaseLoader succeeded: extracted {len(text)} characters")
                return text
        except Exception as e:
            logging.warning(f"WebBaseLoader failed for {url}: {str(e)}")
        
        # Method 2: Try direct download if it's a file URL
        if any(ext in url.lower() for ext in ['.pdf', '.docx', '.zip', '.rar']):
            try:
                logging.info(f"Attempting direct file download for {url}")
                file_type, content = download_file(url)
                if content:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file = os.path.join(temp_dir, f"temp.{file_type}")
                        with open(temp_file, 'wb') as f:
                            f.write(content)
                        
                        if file_type in ['zip', 'rar']:
                            text = process_compressed_file(temp_file, temp_dir)
                        elif file_type == 'pdf':
                            with open(temp_file, 'rb') as f:
                                text = get_pdf_text(f)
                        elif file_type == 'docx':
                            text = get_docx_text(temp_file)
                        
                        if text.strip():
                            logging.info(f"Direct download succeeded: extracted {len(text)} characters")
                            return text
            except Exception as e:
                logging.warning(f"Direct download failed for {url}: {str(e)}")
        
        # Method 3: Try Selenium as last resort (slowest, may fail in Azure)
        try:
            logging.info(f"Attempting SeleniumURLLoader for {url}")
            loader = SeleniumURLLoader(urls=[url])
            docs = loader.load()
            text = "\n".join(doc.page_content for doc in docs)
            if text.strip():
                logging.info(f"SeleniumURLLoader succeeded: extracted {len(text)} characters")
                return text
        except WebDriverException as e:
            logging.error(f"Selenium WebDriver not available: {str(e)}")
        except Exception as e:
            logging.warning(f"SeleniumURLLoader failed for {url}: {str(e)}")
        
        logging.error(f"All methods failed to extract content from {url}")
        return ""
        
    except Exception as e:
        logging.error(f"Comprehensive URL processing error for {url}: {e}", exc_info=True)
        return ""
    
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def cleanup(response):
    gc.collect()
    return response

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={'device': "cpu"}
    )

def get_docx_text(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error processing DOCX: {e}")
        return ""

def get_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        if hasattr(pdf_file, 'content_length') and pdf_file.content_length > MAX_FILE_SIZE:
            return ""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages[:50]:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return ""

def process_file(file):
    """Process uploaded file and return extracted text"""
    if not file or not allowed_file(file.filename):
        return ""
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, file.filename)
            file.save(temp_file)
            
            if file.filename.endswith(('.zip', '.rar')):
                return process_compressed_file(temp_file, temp_dir)
            elif file.filename.endswith('.pdf'):
                with open(temp_file, 'rb') as f:
                    return get_pdf_text(f)

                #     return get_pdf_text(f)
            elif file.filename.endswith('.docx'):
                return get_docx_text(temp_file)
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {e}")
    return ""

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)

def cleanup_old_sessions(max_age_seconds=3600): # Defaults to 1 hour
    """
    STORAGE SAVER: Deletes user folders that haven't been used in 1 hour.
    This prevents your Azure B3 storage from filling up.
    """
    indexes_dir = "faiss_indexes"
    if not os.path.exists(indexes_dir):
        return
    
    current_time = time.time()
    count = 0
    
    for user_id in os.listdir(indexes_dir):
        user_folder = os.path.join(indexes_dir, user_id)
        
        # Skip if it's not a directory
        if not os.path.isdir(user_folder):
            continue
            
        try:
            # Check when the folder was last modified/accessed
            folder_time = os.path.getmtime(user_folder)
            
            # If older than limit, DELETE IT to save space
            if current_time - folder_time > max_age_seconds:
                shutil.rmtree(user_folder)
                count += 1
        except Exception as e:
            logging.warning(f"Failed to cleanup session {user_id}: {e}")
            
    if count > 0:
        logging.info(f"Storage Cleanup: Removed {count} expired user sessions.")

def load_user_vector_store():
    """Helper to safely load the vector store from disk."""
    user_id = session.get('user_id')
    if not user_id:
        return None
    
    folder_path = os.path.join("faiss_indexes", user_id)
    
    if not os.path.exists(folder_path):
        return None
        
    try:
        # STORAGE PROTECTION: 'Touch' the folder to update its timestamp.
        # This tells the cleanup function "I am still here, don't delete me!"
        os.utime(folder_path, None)
        
        embeddings = get_embeddings()
        return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        return None

    
def get_vector_store(text_chunks):
    try:
        user_id = session.get('user_id')
        if not user_id:
            logging.error("No user_id found in session")
            return False

        # 1. TRIGGER CLEANUP: Delete old files before creating new ones
        cleanup_old_sessions()

        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # 2. SAVE TO DISK
        folder_path = os.path.join("faiss_indexes", user_id)
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            
        os.makedirs(folder_path)
        vector_store.save_local(folder_path)
        
        logging.info(f"Vector store saved to disk: {folder_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return False        

@lru_cache(maxsize=1)
def get_qa_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context keeping the tone professional and 
    acting like an expert. If you don't know the answer, just say "Answer is not there within the context", 
    don't provide a wrong answer.Ensure each answer is structured, avoids generic statements, and uses technical terminology where relevant.\n\n
    Context: \n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
def user_ip(user_question, persona):
    try:
        # Load from Disk (this also updates the timestamp so data isn't deleted)
        new_db = load_user_vector_store()
        
        if not new_db:
            return "Please upload documents or process URLs first.", [], None

        docs = new_db.similarity_search(user_question, k=5)
        
        question_types = analyze_question_type(user_question)
        persona_config = get_persona_configuration(persona, question_types)
        system_prompt = build_dynamic_prompt(persona_config, question_types)
        
        prompt = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=GOOGLE_API_KEY,
            temperature=persona_config.get('temperature', 0.3)
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        additional_info = get_additional_info(user_question, persona, question_types)
        
        formatted_response = format_response(
            response["output_text"], 
            persona, 
            question_types
        )
        
        return formatted_response, docs, additional_info
        
    except Exception as e:
        logging.error(f"Error in user_ip: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", [], None
        

def analyze_question_type(question):
    """
    Analyze the question to determine its type for dynamic response formatting.
    """
    question_lower = question.lower()
    
    # Define question patterns
    patterns = {
        'definition': ['what is', 'define', 'meaning of', 'explain'],
        'how_to': ['how to', 'how do', 'how can', 'steps to', 'process of'],
        'why': ['why', 'reason', 'cause', 'purpose'],
        'comparison': ['difference between', 'compare', 'versus', 'vs', 'better than'],
        'example': ['example', 'instance', 'case study', 'illustration'],
        'list': ['list', 'types of', 'kinds of', 'categories'],
        'analysis': ['analyze', 'evaluate', 'assess', 'critique'],
        'application': ['apply', 'use case', 'implement', 'practical'],
        'pros_cons': ['advantages', 'disadvantages', 'pros', 'cons', 'benefits', 'drawbacks'],
        'troubleshooting': ['problem', 'issue', 'error', 'fix', 'solve', 'debug']
    }
    
    detected_types = []
    for qtype, keywords in patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_types.append(qtype)
    
    return detected_types if detected_types else ['general']


def get_persona_configuration(persona, question_types):
    """
    Get dynamic configuration for each persona based on question type.
    """
    configs = {
        "Student": {
            'base_instruction': "You are explaining to a student who is learning this topic.",
            'temperature': 0.4,
            'format_guides': {
                'definition': "Start with a simple definition, then elaborate with analogies and real-world examples.",
                'how_to': "Break down into clear numbered steps with explanations for each step.",
                'why': "Explain the reasoning in a logical flow, connecting cause and effect clearly.",
                'comparison': "Create a clear comparison showing similarities first, then differences.",
                'example': "Provide multiple relatable examples from everyday life.",
                'list': "Present as an organized list with brief explanations for each item.",
                'analysis': "Guide through the analysis step-by-step, teaching the thought process.",
                'application': "Show practical applications with step-by-step implementation.",
                'pros_cons': "Present in a balanced way with equal weight to both sides.",
                'troubleshooting': "Walk through the problem-solving process pedagogically.",
                'general': "Explain concepts clearly with examples and build understanding progressively."
            },
            'style_notes': [
                "Use simple language and avoid unnecessary jargon",
                "Include analogies and metaphors to aid understanding",
                "Provide concrete examples from familiar contexts",
                "Encourage learning by explaining the 'why' behind concepts",
                "Structure information in digestible chunks"
            ]
        },
        
        "Researcher": {
            'base_instruction': "You are addressing an academic researcher or scholar.",
            'temperature': 0.2,
            'format_guides': {
                'definition': "Provide precise technical definition with theoretical foundations and academic context.",
                'how_to': "Detail methodology with theoretical justification and research considerations.",
                'why': "Analyze underlying mechanisms with references to theoretical frameworks.",
                'comparison': "Conduct systematic comparison with critical analysis of methodological approaches.",
                'example': "Present case studies with methodological details and research implications.",
                'list': "Categorize systematically with theoretical basis for classification.",
                'analysis': "Perform rigorous analysis with attention to validity, limitations, and research gaps.",
                'application': "Discuss research applications with methodological considerations.",
                'pros_cons': "Evaluate critically with attention to research quality and evidence base.",
                'troubleshooting': "Address methodological challenges with research-based solutions.",
                'general': "Provide comprehensive academic treatment with theoretical depth."
            },
            'style_notes': [
                "Use precise technical terminology",
                "Reference theoretical frameworks and research paradigms",
                "Discuss methodological considerations",
                "Address limitations and gaps in current knowledge",
                "Maintain scholarly rigor and objectivity"
            ]
        },
        
        "Working Professional": {
            'base_instruction': "You are advising a working professional seeking practical knowledge.",
            'temperature': 0.3,
            'format_guides': {
                'definition': "Define in business context with relevance to professional practice.",
                'how_to': "Provide actionable steps with time estimates and resource requirements.",
                'why': "Explain business rationale with focus on ROI and practical impact.",
                'comparison': "Compare with focus on practical implications and decision-making criteria.",
                'example': "Share industry examples and real-world case studies.",
                'list': "Prioritize by relevance to professional practice.",
                'analysis': "Analyze with focus on actionable insights and business implications.",
                'application': "Detail implementation with consideration for organizational context.",
                'pros_cons': "Frame in terms of business value and practical considerations.",
                'troubleshooting': "Provide practical solutions with implementation guidance.",
                'general': "Focus on actionable insights and real-world application."
            },
            'style_notes': [
                "Emphasize practical application and business value",
                "Include time and resource considerations",
                "Reference industry standards and best practices",
                "Focus on actionable takeaways",
                "Consider organizational and professional context"
            ]
        },
        
        "Teacher": {
            'base_instruction': "You are helping a teacher prepare lesson content or understand pedagogical approaches.",
            'temperature': 0.35,
            'format_guides': {
                'definition': "Structure as you would teach it, with scaffolded explanation and formative check points.",
                'how_to': "Present as a lesson plan with learning objectives, activities, and assessment strategies.",
                'why': "Explain in ways that help students build conceptual understanding.",
                'comparison': "Create comparative framework suitable for classroom instruction.",
                'example': "Provide teaching examples with pedagogical notes on usage.",
                'list': "Organize for progressive learning with curriculum alignment.",
                'analysis': "Model analytical thinking for classroom demonstration.",
                'application': "Suggest classroom activities and student exercises.",
                'pros_cons': "Frame for classroom discussion with guiding questions.",
                'troubleshooting': "Address common student misconceptions and teaching challenges.",
                'general': "Structure content for effective classroom delivery with pedagogical guidance."
            },
            'style_notes': [
                "Include learning objectives and outcomes",
                "Suggest teaching strategies and activities",
                "Address common student misconceptions",
                "Provide formative assessment ideas",
                "Consider differentiation for diverse learners",
                "Include discussion prompts and guiding questions"
            ]
        },
        
        "Product Manager": {
            'base_instruction': "You are advising a product manager on product strategy and execution.",
            'temperature': 0.35,
            'format_guides': {
                'definition': "Define with focus on user value and product implications.",
                'how_to': "Outline execution roadmap with milestones and success metrics.",
                'why': "Explain in terms of user needs, business goals, and market dynamics.",
                'comparison': "Compare competitive positioning and strategic advantages.",
                'example': "Provide product case studies with feature analysis.",
                'list': "Prioritize using product frameworks (RICE, MoSCoW, etc.).",
                'analysis': "Analyze user impact, business metrics, and technical feasibility.",
                'application': "Detail implementation with go-to-market considerations.",
                'pros_cons': "Evaluate trade-offs with user and business impact.",
                'troubleshooting': "Address product challenges with user-centric solutions.",
                'general': "Frame in terms of user value, business impact, and product success."
            },
            'style_notes': [
                "Focus on user value and business outcomes",
                "Include metrics and success criteria",
                "Consider technical feasibility and scalability",
                "Reference product frameworks and methodologies",
                "Address stakeholder perspectives"
            ]
        },
        
        "Startup Founder": {
            'base_instruction': "You are advising a startup founder on strategy and execution.",
            'temperature': 0.4,
            'format_guides': {
                'definition': "Define with focus on market opportunity and competitive advantage.",
                'how_to': "Provide lean execution strategy with validation milestones.",
                'why': "Explain market dynamics, timing, and strategic positioning.",
                'comparison': "Analyze competitive landscape and differentiation opportunities.",
                'example': "Share startup case studies with growth insights.",
                'list': "Prioritize by impact and resource efficiency.",
                'analysis': "Evaluate market fit, scalability, and growth potential.",
                'application': "Detail MVP approach with iteration strategy.",
                'pros_cons': "Assess in terms of market opportunity and execution risk.",
                'troubleshooting': "Address startup challenges with scrappy solutions.",
                'general': "Focus on innovation, market opportunity, and execution excellence."
            },
            'style_notes': [
                "Emphasize speed, innovation, and market opportunity",
                "Consider resource constraints and efficiency",
                "Focus on competitive differentiation",
                "Include growth and scaling considerations",
                "Address execution risks and mitigation"
            ]
        },
        
        "Developer": {
            'base_instruction': "You are providing technical guidance to a software developer.",
            'temperature': 0.25,
            'format_guides': {
                'definition': "Provide technical definition with implementation details.",
                'how_to': "Detail implementation steps with code considerations and best practices.",
                'why': "Explain technical rationale with architecture and design considerations.",
                'comparison': "Compare technical approaches with performance and maintainability analysis.",
                'example': "Provide code examples with detailed explanations.",
                'list': "Organize by technical category with usage context.",
                'analysis': "Analyze technical trade-offs, performance, and scalability.",
                'application': "Detail implementation with code structure and patterns.",
                'pros_cons': "Evaluate technical merits with performance implications.",
                'troubleshooting': "Debug systematically with technical diagnostics.",
                'general': "Provide technical depth with implementation guidance."
            },
            'style_notes': [
                "Use precise technical terminology",
                "Include code-level insights where applicable",
                "Discuss performance and scalability",
                "Reference design patterns and best practices",
                "Consider maintainability and code quality"
            ]
        },
        
        "Policy Maker": {
            'base_instruction': "You are advising a policy maker on regulatory and societal implications.",
            'temperature': 0.3,
            'format_guides': {
                'definition': "Define with attention to regulatory context and public impact.",
                'how_to': "Outline policy framework with implementation and enforcement considerations.",
                'why': "Explain societal impact, ethical implications, and public interest.",
                'comparison': "Compare policy approaches with equity and effectiveness analysis.",
                'example': "Provide policy case studies with outcome analysis.",
                'list': "Prioritize by public impact and policy objectives.",
                'analysis': "Evaluate social equity, ethical implications, and long-term effects.",
                'application': "Detail policy implementation with stakeholder considerations.",
                'pros_cons': "Assess public benefit, equity, and unintended consequences.",
                'troubleshooting': "Address policy challenges with inclusive solutions.",
                'general': "Focus on public benefit, equity, and regulatory implications."
            },
            'style_notes': [
                "Consider regulatory and ethical dimensions",
                "Address equity and accessibility",
                "Discuss societal and long-term impacts",
                "Include stakeholder perspectives",
                "Reference policy frameworks and precedents"
            ]
        },
        
        "Investor": {
            'base_instruction': "You are providing investment analysis and market insights.",
            'temperature': 0.3,
            'format_guides': {
                'definition': "Define with focus on market size and investment opportunity.",
                'how_to': "Outline investment thesis with due diligence considerations.",
                'why': "Explain market dynamics, trends, and investment rationale.",
                'comparison': "Compare investment opportunities with risk-return analysis.",
                'example': "Provide market examples with financial performance.",
                'list': "Prioritize by investment potential and market opportunity.",
                'analysis': "Evaluate market potential, business model, and financial viability.",
                'application': "Detail go-to-market with monetization strategy.",
                'pros_cons': "Assess investment opportunity against market and execution risks.",
                'troubleshooting': "Address business challenges with strategic solutions.",
                'general': "Focus on ROI, market potential, and business model viability."
            },
            'style_notes': [
                "Focus on financial metrics and ROI",
                "Analyze market size and growth potential",
                "Evaluate business model and monetization",
                "Assess competitive landscape and moat",
                "Consider risk factors and mitigation"
            ]
        }
    }
    
    config = configs.get(persona, configs["Student"])
    
    # Select appropriate format guide based on question types
    primary_type = question_types[0] if question_types else 'general'
    config['active_format_guide'] = config['format_guides'].get(primary_type, config['format_guides']['general'])
    
    return config


def build_dynamic_prompt(persona_config, question_types):
    """
    Build a dynamic prompt based on persona configuration and question type.
    """
    base = persona_config['base_instruction']
    format_guide = persona_config['active_format_guide']
    style_notes = "\n".join(f"- {note}" for note in persona_config['style_notes'])
    
    prompt = f"""
{base}

RESPONSE FORMATTING GUIDELINE:
{format_guide}

STYLE REQUIREMENTS:
{style_notes}

IMPORTANT INSTRUCTIONS:
- Use ONLY the information from the provided context to answer the question
- If the answer is not in the context, respond with: "Answer is not there within the context."
- Do NOT make up information or use external knowledge
- Maintain the specified persona perspective throughout
- Adapt your explanation style to the type of question being asked
- Be concise but thorough - avoid unnecessary verbosity
- Use clear structure and logical flow

Context: {{context}}

Question: {{question}}

Answer:
"""
    return prompt


def format_response(response_text, persona, question_types):
    """
    Format and clean the response based on persona and question type.
    """
    # Remove markdown formatting
    cleaned = response_text
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove italic
    cleaned = re.sub(r'^[-•]\s+', '', cleaned, flags=re.MULTILINE)  # Remove bullets
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)        # Remove excessive newlines
    
    # Persona-specific formatting enhancements
    if persona == "Teacher":
        # Add subtle structure markers for teachers
        if any(qtype in question_types for qtype in ['how_to', 'definition']):
            # Ensure logical flow is maintained
            cleaned = enhance_pedagogical_structure(cleaned)
    
    elif persona == "Developer":
        # Preserve technical formatting
        cleaned = preserve_code_references(cleaned)
    
    elif persona == "Researcher":
        # Maintain academic rigor markers
        cleaned = enhance_academic_structure(cleaned)
    
    return cleaned.strip()


def enhance_pedagogical_structure(text):
    """
    Add subtle pedagogical structure to teacher responses.
    """
    # This maintains readability while ensuring teaching flow
    return text


def preserve_code_references(text):
    """
    Preserve code-related formatting for developer persona.
    """
    # Keep technical terms clear
    return text


def enhance_academic_structure(text):
    """
    Maintain academic structure for researcher persona.
    """
    # Preserve scholarly formatting
    return text


def get_additional_info(query, persona=None, question_types=None):
    """
    Get persona-aware additional information from Gemini.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Persona-aware prompting for additional info
        persona_context = ""
        if persona:
            persona_styles = {
                "Student": "in a beginner-friendly way with practical examples",
                "Researcher": "with academic depth and theoretical context",
                "Working Professional": "with actionable insights and business relevance",
                "Teacher": "with pedagogical value and teaching applications",
                "Product Manager": "with product strategy and user value focus",
                "Startup Founder": "with market opportunity and competitive insights",
                "Developer": "with technical implementation details",
                "Policy Maker": "with regulatory and societal implications",
                "Investor": "with market potential and investment perspective"
            }
            persona_context = persona_styles.get(persona, "")
        
        enhanced_prompt = f"""
        Provide additional relevant information {persona_context} about the following topic.
        
        Topic: {query}
        
        Cover complementary aspects such as:
        - Recent developments or trends
        - Practical applications or use cases
        - Related concepts or technologies
        - Best practices or expert insights
        
        Write in clear, natural language paragraphs without markdown formatting, bullet points, 
        or special characters. Focus on information that complements what might already be 
        covered in the main response.
        """
        
        response = model.generate_content(enhanced_prompt)
        return response.text.strip()
        
    except Exception as e:
        logging.error(f"Error getting additional information: {e}")
        return None

        
def generate_common_questions(docs):
    try:
        prompt = """you are an professional in generating good and applicable questions. Generate 5 questions according to the file and if the questions are not enough 
        then generate questions relevant to the context.
        
        Document content: {context}
        
        Generate 7 important questions:"""
        
        chain = get_qa_chain()
        response = chain(
            {'input_documents': docs, 'question': prompt},
            return_only_outputs=True
        )
        
        # Extract questions from the response
        questions = [q.strip() for q in response['output_text'].split('\n') if q.strip()]
        return questions[:7]  # Ensure we return max 5 questions
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return []

def generate_key_concepts(docs):
    try:
        prompt = """Given the following document content, identify and list the 5 most important key concepts or main ideas. Format them as an in-depth list.
        
        Document content: {context}
        
        Generate 5 key concepts:"""
        
        chain = get_qa_chain()
        response = chain(
            {'input_documents': docs, 'question': prompt},
            return_only_outputs=True
        )
        
        # Extract concepts from the response
        concepts = [c.strip() for c in response['output_text'].split('\n') if c.strip()]
        return concepts[:5]  # Ensure we return max 5 concepts
    except Exception as e:
        logging.error(f"Error generating concepts: {e}")
        return []


@lru_cache(maxsize=10)
def get_video_recommendations(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(
            f'https://www.youtube.com/results?search_query={"+".join(query.split())}',
            headers=headers,
            timeout=10
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if 'var ytInitialData' in str(script.string or ''):
                data = json.loads(script.string[script.string.find('{'):script.string.rfind('}')+1])
                contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [{}])[0].get('itemSectionRenderer', {}).get('contents', [])
                video_ids = [i.get('videoRenderer', {}).get('videoId') for i in contents if 'videoRenderer' in i][:MAX_VIDEOS]
                return [
                    {
                        "video_id": vid,
                        "thumbnail_url": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
                    }
                    for vid in video_ids if vid
                ]
        return []
    except Exception:
        return []

@app.route("/")
def index():
    return "Backend is running"

@app.route("/documentation", methods=["GET"])
def documentation():
    return "https://docdy-documentation.vercel.app/"

@app.route("/privacy-policy", methods=["GET"])
def privacy_policy():
    return "https://docdynamo-privacy.vercel.app/"


@app.route('/process_urls', methods=['POST'])
def process_urls():
    try:
        urls = request.json.get('urls', [])
        if not urls:
            return jsonify({
                'success': False,
                'error': 'No URLs provided'
            }), 400
        
        processed_urls = []
        failed_urls = []
        all_text = ""
        
        for url in urls:
            if not is_valid_url(url):
                failed_urls.append({'url': url, 'reason': 'Invalid URL format'})
                continue
                
            try:
                text = process_url_file(url)
                if text.strip():
                    all_text += text + "\n"
                    processed_urls.append(url)
                    logging.info(f"Successfully processed URL: {url} - extracted {len(text)} characters")
                else:
                    failed_urls.append({'url': url, 'reason': 'No content extracted'})
                    logging.warning(f"No content extracted from URL: {url}")
            except Exception as e:
                failed_urls.append({'url': url, 'reason': str(e)})
                logging.error(f"Failed to process URL {url}: {str(e)}")
        
        if all_text.strip():
            # Create text chunks
            text_chunks = get_chunks(all_text)
            logging.info(f"Created {len(text_chunks)} text chunks from URLs")
            
            # Save to Memory (RAM) via user_sessions
            success = get_vector_store(text_chunks)
            
            if not success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create memory index',
                    'details': {
                        'processed_urls': processed_urls,
                        'failed_urls': failed_urls
                    }
                }), 500
            
            # *** CRITICAL FIX: Set session flags ***
            session['urls_processed'] = True
            session['uploaded_urls'] = processed_urls
            session['has_documents'] = True  # Add this flag
            session.modified = True  # Ensure session is saved
            
            logging.info(f"Successfully processed {len(processed_urls)} URLs and created vector store")
            
            return jsonify({
                'success': True,
                'processed_urls': processed_urls,
                'failed_urls': failed_urls,
                'chunks_created': len(text_chunks),
                'total_characters': len(all_text)
            })
        
        # Detailed error message if no content was extracted
        return jsonify({
            'success': False,
            'error': 'Could not extract content from any URLs',
            'details': {
                'failed_urls': failed_urls,
                'attempted_urls': urls,
                'message': 'No content could be extracted from the provided URLs. This may be due to: 1) URLs requiring authentication, 2) JavaScript-heavy pages, 3) Anti-scraping protections'
            }
        }), 400
        
    except Exception as e:
        logging.error(f"Error processing URLs: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'details': {
                'type': type(e).__name__,
                'message': str(e)
            }
        }), 500

@app.route("/api/generate", methods=["POST"])
def generate_mindmap_api():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing input text"}), 400

    mindmap_md = create_mindmap_markdown(text)
    return jsonify({"mindmap_markdown": mindmap_md})

   
# Add new route for MD/PDF download

@app.route('/download_mindmap_md', methods=['GET'])
def download_mindmap_md():
    try:
        md_path = "temp/mindmap.md"
        if os.path.exists(md_path):
            return send_file(
                md_path,
                as_attachment=True,
                download_name="DocDynamo_Mindmap.md",
                mimetype="text/markdown"
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Markdown file not found'
            }), 404
    except Exception as e:
        logging.error(f"Error downloading Markdown: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/download_pdf", methods=["GET"])
def download_pdf_api():
    return send_file("static/generated/output.pdf", as_attachment=True)
    

@app.route('/start_over', methods=['POST'])
def start_over():
    user_id = session.get('user_id')
    
    # Immediate Cleanup for this user
    if user_id:
        folder_path = os.path.join("faiss_indexes", user_id)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                logging.error(f"Error removing user folder: {e}")

    session.clear()
    session['user_id'] = str(uuid.uuid4())
    
    return jsonify({'success': True, 'message': 'Session cleared successfully'})    

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        new_db = load_user_vector_store()
        
        if not new_db:
            return jsonify({'success': False, 'error': 'No documents uploaded'}), 400

        docs = new_db.similarity_search("", k=3)
        questions = generate_common_questions(docs)
        session['generated_questions'] = questions
        
        return jsonify({'success': True, 'questions': questions})
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate_concepts', methods=['POST'])
def generate_concepts():
    try:
        new_db = load_user_vector_store()
        
        if not new_db:
            return jsonify({'success': False, 'error': 'No documents uploaded'}), 400

        docs = new_db.similarity_search("", k=3)
        concepts = generate_key_concepts(docs)
        session['key_concepts'] = concepts
        
        return jsonify({'success': True, 'concepts': concepts})
    except Exception as e:
        logging.error(f"Error generating concepts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_additional_info', methods=['POST'])
def get_additional_info_route():
    try:
        new_db = load_user_vector_store()
        
        if not new_db:
            return jsonify({'success': False, 'error': 'No documents uploaded'}), 400

        docs = new_db.similarity_search("", k=5)
        context = "\n".join(doc.page_content for doc in docs)
        additional_info = get_additional_info(context)
        
        if additional_info:
            return jsonify({'success': True, 'additional_info': additional_info})
        else:
            return jsonify({'success': False, 'error': 'Could not generate info'}), 400
            
    except Exception as e:
        logging.error(f"Error getting additional information: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
        
    
# Replace the index route with this fixed version:
@app.route('/api/query', methods=['POST'])
def android_query():
    response = None
    additional_info = None
    recommendations = []
    uploaded_filenames = []

    user_question = request.form.get('question', '').strip()
    persona = request.form.get('persona', 'Student')
    files = request.files.getlist('docs')

    # Handle Uploads
    if files and any(file.filename for file in files):
        all_text = ""
        for file in files:
            if file and allowed_file(file.filename):
                text = process_file(file)
                all_text += text + "\n"
                uploaded_filenames.append(file.filename)

        if all_text.strip():
            text_chunks = get_chunks(all_text)
            # Calls get_vector_store which now triggers auto-cleanup
            success = get_vector_store(text_chunks)
            if not success:
                return jsonify({
                    "response": "Failed to create knowledge base.",
                    "additional_info": None,
                    "recommendations": [],
                    "uploaded_filenames": uploaded_filenames
                }), 500
            session['has_documents'] = True
            session.modified = True

    # Handle Question
    if user_question:
        new_db = load_user_vector_store()
        
        if not new_db:
             return jsonify({
                "response": "Please upload documents or process URLs first.",
                "additional_info": None,
                "recommendations": [],
                "uploaded_filenames": uploaded_filenames
            }), 400
            
        response, docs, additional_info = user_ip(user_question, persona)
        if response and docs:
            context_text = " ".join(doc.page_content for doc in docs)
            video_query = f"{response} {context_text}".strip()
            recommendations = get_video_recommendations(video_query)

    return jsonify({    
        "response": response,
        "additional_info": additional_info,
        "recommendations": recommendations,
        "uploaded_filenames": uploaded_filenames
    })


if __name__ == '__main__':
     app.run(debug=os.getenv("FLASK_DEBUG", False), threaded=True, host="0.0.0.0")





