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

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)
CORS(app) 
app.secret_key = os.getenv('FLASK_SECRET_KEY', "6fK9P6WcfpBz7bWJ9qV2eP2Qv5dA8D8z")

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

@app.route('/get_additional_info', methods=['POST'])
def get_additional_info_route():
    try:
        if not os.path.exists("faiss_index"):
            return jsonify({
                'success': False,
                'error': 'No documents have been uploaded yet'
            }), 400

        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("", k=5)  # Get representative documents
        
        # Get a summary of the document content
        context = "\n".join(doc.page_content for doc in docs)
        additional_info = get_additional_info(context)
        
        if additional_info:
            return jsonify({
                'success': True,
                'additional_info': additional_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate additional information'
            }), 400
            
    except Exception as e:
        logging.error(f"Error getting additional information: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        # First, try WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
        
        # If no text, try Selenium
        if not text.strip():
            loader = SeleniumURLLoader(urls=[url])
            docs = loader.load()
            text = "\n".join(doc.page_content for doc in docs)
        
        # If still no text, try direct download and file processing
        if not text.strip():
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
        
        return text or ""
        
    except Exception as e:
        logging.error(f"Comprehensive URL processing error: {e}")
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

def get_vector_store(text_chunks):
    try:
        # Create the directory if it doesn't exist
        os.makedirs("faiss_index", exist_ok=True)
        
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        
        # Verify the index was created successfully
        if not os.path.exists("faiss_index/index.faiss"):
            logging.error("Failed to create FAISS index file")
            return False
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

def get_additional_info(query):
    """Get additional information from Gemini for the query"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Craft a prompt that encourages complementary information
        enhanced_prompt = f"""
        You are a helpful assistant. Based on the following content, provide additional relevant information in plain text with no markdown symbols, no bullet points, and no formatting characters like '*', '-', or '**'.
        Topic:
        {query}
        Cover the following in plain text:
        1. Recent developments or updates
        2. Common practical applications
        3. Related concepts or technologies
        4. Expert insights or best practices
        Write clearly and concisely in paragraph format.
        """

        
        response = model.generate_content(enhanced_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error getting additional information: {e}")
        return None
    
    
def user_ip(user_question, persona):
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)

        persona_instructions = {
            "Student": "Explain in a simple, beginner-friendly way with relatable examples.",
            "Researcher": "Provide a technically deep explanation with references and advanced insights.",
            "Working Professional": "Focus on practicality, real-world application, and relevance to industry.",
            "Teacher": "Structure the answer clearly like a lesson plan or explanation for a classroom.",
            "Product Manager": "Frame the answer in terms of user value, business impact, and scalability.",
            "Startup Founder": "Highlight innovation, execution strategy, and competitive advantages.",
            "Developer": "Provide code-level insights, examples, and technical clarity.",
            "Policy Maker": "Consider regulatory and ethical implications with broader societal context.",
            "Investor": "Discuss ROI, market potential, business model implications, and trends."
        }

        system_prompt = f"""
        {persona_instructions.get(persona, '')}

        Use the context below to answer the question. If the answer isn't in the context, say:
        "Answer is not there within the context."

        Context: {{context}}
        Question: {{question}}
        Answer:
        """

        prompt = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY,
                                     temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        additional_info = get_additional_info(user_question)

        cleaned_response = response["output_text"]
        cleaned_response = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_response)  # remove **bold**
        cleaned_response = re.sub(r'\*(.*?)\*', r'\1', cleaned_response)      # remove *italic*
        cleaned_response = re.sub(r'^[-•]\s+', '', cleaned_response, flags=re.MULTILINE)  # remove - or • at line start
        cleaned_response = re.sub(r'\n{2,}', '\n\n', cleaned_response)        # remove extra newlines
        
        return cleaned_response.strip(), docs, additional_info

    except Exception as e:
        return f"Error: {str(e)}", [], None

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

def verify_faiss_index():
    """Verify that the FAISS index exists and is valid"""
    if not os.path.exists("faiss_index/index.faiss"):
        return False
    try:
        embeddings = get_embeddings()
        FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        return True
    except Exception as e:
        logging.error(f"Error verifying FAISS index: {e}")
        return False



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
                else:
                    failed_urls.append({'url': url, 'reason': 'No content extracted'})
            except Exception as e:
                failed_urls.append({'url': url, 'reason': str(e)})
        
        if all_text.strip():
            # Create the directory if it doesn't exist
            os.makedirs("faiss_index", exist_ok=True)
            
            text_chunks = get_chunks(all_text)
            get_vector_store(text_chunks)
            
            # Verify the index was created successfully
            if not os.path.exists("faiss_index/index.faiss"):
                return jsonify({
                    'success': False,
                    'error': 'Failed to create vector index',
                    'details': {
                        'processed_urls': processed_urls,
                        'failed_urls': failed_urls
                    }
                }), 500
                
            session['uploaded_urls'] = processed_urls
            return jsonify({
                'success': True,
                'processed_urls': processed_urls,
                'failed_urls': failed_urls
            })
        
        # Detailed error message if no content was extracted
        return jsonify({
            'success': False,
            'error': 'Could not process URLs',
            'details': {
                'failed_urls': failed_urls,
                'attempted_urls': urls,
                'message': 'No content could be extracted from the provided URLs'
            }
        }), 400
        
    except Exception as e:
        logging.error(f"Error processing URLs: {e}")
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
    session.clear()
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
        except Exception as e:
            logging.error(f"Error removing faiss_index directory: {e}")
    return redirect(url_for('index'))


@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        if not os.path.exists("faiss_index/index.faiss"):
            return jsonify({
                'success': False,
                'error': 'No documents have been uploaded yet'
            }), 400

        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("", k=3)  # Get representative documents
        
        questions = generate_common_questions(docs)
        session['generated_questions'] = questions
        
        return jsonify({
            'success': True,
            'questions': questions
        })
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/generate_concepts', methods=['POST'])
def generate_concepts():
    try:
        if not os.path.exists("faiss_index/index.faiss"):
            return jsonify({
                'success': False,
                'error': 'No documents have been uploaded yet'
            }), 400

        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("", k=3)  # Get representative documents
        
        concepts = generate_key_concepts(docs)
        session['key_concepts'] = concepts
        
        return jsonify({
            'success': True,
            'concepts': concepts
        })
    except Exception as e:
        logging.error(f"Error generating concepts: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    
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

    if files and any(file.filename for file in files):
        all_text = ""
        for file in files:
            if file and allowed_file(file.filename):
                text = process_file(file)
                all_text += text + "\n"
                uploaded_filenames.append(file.filename)

        if all_text.strip():
            text_chunks = get_chunks(all_text)
            success = get_vector_store(text_chunks)
            if not success:
                return jsonify({
                    "response": "Failed to create knowledge base from uploaded documents.",
                    "additional_info": None,
                    "recommendations": [],
                    "uploaded_filenames": uploaded_filenames
                }), 500

    if user_question:
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



