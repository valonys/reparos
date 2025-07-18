"""
Utilities module for DigiTwin Analytics
Contains common functions, decorators, and data processing utilities
"""

import logging
import pandas as pd
from functools import wraps
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import streamlit as st
from config import (
    NI_keywords, NC_keywords, module_keywords, rack_keywords, 
    living_quarters_keywords, flare_keywords, fwd_keywords, hexagons_keywords,
    NI_keyword_map, NC_keyword_map
)

import matplotlib.patches as patches
import math
import matplotlib.transforms as transforms

# PAZ-specific keywords for data processing
paz_module_keywords = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
paz_rack_keywords = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

# PAZ keyword mapping for preprocessing
paz_keyword_map = {
    'P1': 'P1', 'P2': 'P2', 'P3': 'P3', 'P4': 'P4', 'P5': 'P5', 'P6': 'P6', 'P7': 'P7', 'P8': 'P8',
    'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S4': 'S4', 'S5': 'S5', 'S6': 'S6', 'S7': 'S7', 'S8': 'S8',
    'R1': 'R1', 'R2': 'R2', 'R3': 'R3', 'R4': 'R4', 'R5': 'R5', 'R6': 'R6'
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DECORATORS ---
def log_execution(func):
    """Decorator to log function execution for debugging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

# --- DATA PROCESSING FUNCTIONS ---
@log_execution
def parse_pdf(file):
    """Parse PDF file and extract text content"""
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

@st.cache_resource
def build_faiss_vectorstore(_docs):
    """Build FAISS vectorstore from documents with caching"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(_docs):
        for chunk in splitter.split_text(doc.page_content):
            chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
    return FAISS.from_documents(chunks, embeddings)

@log_execution
def preprocess_keywords(description):
    """Preprocess description text for keyword extraction"""
    description = str(description).upper()
    for lq_variant in living_quarters_keywords:
        if lq_variant != 'LQ':
            description = description.replace(lq_variant, 'LQ')
    
    # Handle CLV module keywords
    for module in module_keywords:
        number = module[1:]
        if number in description:
            description = description.replace(number, module)
    
    # Handle PAZ module keywords
    for module in paz_module_keywords:
        if module in description:
            description = description.replace(module, module)
    
    # Handle PAZ rack keywords
    for rack in paz_rack_keywords:
        if rack in description:
            description = description.replace(rack, rack)
    
    for original, grouped in {**NI_keyword_map, **NC_keyword_map}.items():
        description = description.replace(original, grouped)
    return description

@log_execution
def extract_ni_nc_keywords(row, notif_type_col, desc_col):
    """Extract NI/NC keywords from notification row"""
    description = preprocess_keywords(row[desc_col])
    notif_type = row[notif_type_col]
    keywords = [kw for kw in (NI_keywords if notif_type == 'NI' else NC_keywords) if kw in description]
    return ', '.join(keywords) if keywords else 'None'

@log_execution
def extract_location_keywords(row, desc_col, keyword_list):
    """Extract location keywords from notification row"""
    description = preprocess_keywords(row[desc_col])
    if keyword_list == living_quarters_keywords:
        return 'LQ' if any(kw in description for kw in living_quarters_keywords) else 'None'
    locations = [kw for kw in keyword_list if kw in description]
    return ', '.join(locations) if locations else 'None'

@log_execution
def create_pivot_table(df, index, columns, aggfunc='size', fill_value=0):
    """Create pivot table from dataframe"""
    df_exploded = df.assign(Keywords=df[columns].str.split(', ')).explode('Keywords')
    df_exploded = df_exploded[df_exploded['Keywords'] != 'None']
    pivot = pd.pivot_table(df_exploded, index=index, columns='Keywords', aggfunc=aggfunc, fill_value=fill_value)
    return pivot

@log_execution
def apply_fpso_colors(df):
    """Apply color styling to FPSO dataframe"""
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    color_map = {'GIR': '#FFA07A', 'DAL': '#ADD8E6', 'PAZ': '#D8BFD8', 'CLV': '#90EE90'}
    for fpso, color in color_map.items():
        if fpso in df.index:
            styles.loc[fpso] = f'background-color: {color}'
    return styles

@log_execution
def process_uploaded_files(files):
    """Process uploaded files and return PDF documents and Excel dataframe"""
    pdf_files = [f for f in files if f.type == "application/pdf"]
    excel_files = [f for f in files if f.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    
    # Process PDF files
    parsed_docs = []
    if pdf_files:
        parsed_docs = [LCDocument(page_content=parse_pdf(f), metadata={"name": f.name}) for f in pdf_files]
        st.sidebar.success(f"{len(parsed_docs)} PDF reports indexed.")
    
    # Process Excel files
    df = None
    if excel_files:
        try:
            # Use the first Excel file if multiple are uploaded
            uploaded_xlsx = excel_files[0]
            df = pd.read_excel(uploaded_xlsx, sheet_name='Global Notifications')
            df.columns = df.columns.str.strip()
            expected_columns = {
                'Notifictn type': 'Notifictn type',
                'Created on': 'Created on',
                'Description': 'Description',
                'FPSO': 'FPSO'
            }
            missing_columns = [col for col in expected_columns.values() if col not in df.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                return parsed_docs, None
            
            df = df[list(expected_columns.values())]
            df.columns = list(expected_columns.keys())
            df = df[df['FPSO'].isin(['GIR', 'DAL', 'PAZ', 'CLV'])]
            df['Extracted_Keywords'] = df.apply(extract_ni_nc_keywords, axis=1, args=('Notifictn type', 'Description'))
            for loc_type, keywords in [
                ('Modules', module_keywords + paz_module_keywords), ('Racks', rack_keywords + paz_rack_keywords), ('LivingQuarters', living_quarters_keywords),
                ('Flare', flare_keywords), ('FWD', fwd_keywords), ('HeliDeck', hexagons_keywords)
            ]:
                df[f'Extracted_{loc_type}'] = df.apply(extract_location_keywords, axis=1, args=('Description', keywords))
            st.sidebar.success("Excel file processed successfully.")
        except Exception as e:
            st.error(f"Error processing Excel: {e}")
            return parsed_docs, None
    
    return parsed_docs, df 

def add_rectangle(ax, xy, width, height, **kwargs):
    rectangle = patches.Rectangle(xy, width, height, **kwargs)
    ax.add_patch(rectangle)

def add_chamfered_rectangle(ax, xy, width, height, chamfer, **kwargs):
    x, y = xy
    coords = [
        (x + chamfer, y),
        (x + width - chamfer, y),
        (x + width, y + chamfer),
        (x + width, y + height - chamfer),
        (x + width - chamfer, y + height),
        (x + chamfer, y + height),
        (x, y + height - chamfer),
        (x, y + chamfer)
    ]
    polygon = patches.Polygon(coords, closed=True, **kwargs)
    ax.add_patch(polygon)

def add_hexagon(ax, xy, radius, **kwargs):
    x, y = xy
    vertices = [(x + radius * math.cos(2 * math.pi * n / 6), y + radius * math.sin(2 * math.pi * n / 6)) for n in range(6)]
    hexagon = patches.Polygon(vertices, closed=True, **kwargs)
    ax.add_patch(hexagon)

def add_fwd(ax, xy, width, height, **kwargs):
    x, y = xy
    top_width = width * 0.80
    coords = [
        (0, 0),
        (width, 0),
        (width - (width - top_width) / 2, height),
        ((width - top_width) / 2, height)
    ]
    trapezoid = patches.Polygon(coords, closed=True, **kwargs)
    t = transforms.Affine2D().rotate_deg(90).translate(x, y)
    trapezoid.set_transform(t + ax.transData)
    ax.add_patch(trapezoid)
    text_t = transforms.Affine2D().rotate_deg(90).translate(x + height / 2, y + width / 2)
    ax.text(0, -1, "FWD", ha='center', va='center', fontsize=7, weight='bold', transform=text_t + ax.transData) 