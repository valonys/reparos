import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import matplotlib.transforms as transforms
import sqlite3

# Import FPSO-specific modules
from clv import *
from paz import *
from dal import *
from gir import *
# Import shared utilities
# Remove these imports:
# from utils import preprocess_keywords, extract_ni_nc_keywords, extract_location_keywords

# --- FAST LOCAL PREPROCESSING FUNCTIONS ---
def preprocess_keywords(description):
    description = str(description).upper()
    for lq_variant in clv_living_quarters_keywords:
        if lq_variant != 'LQ':
            description = description.replace(lq_variant, 'LQ')
    for module in clv_module_keywords:
        number = module[1:]
        if number in description:
            description = description.replace(number, module)
    for module in paz_module_keywords:
        if module in description:
            description = description.replace(module, module)
    for rack in paz_rack_keywords:
        if rack in description:
            description = description.replace(rack, rack)
    for module in dal_module_keywords:
        if module in description:
            description = description.replace(module, module)
    for rack in dal_rack_keywords:
        if rack in description:
            description = description.replace(rack, rack)
    # If you use NI_keyword_map and NC_keyword_map, add them here as well
    return description

def extract_ni_nc_keywords(row, notif_type_col, desc_col):
    description = preprocess_keywords(row[desc_col])
    notif_type = row[notif_type_col]
    if notif_type == 'NI':
        keywords = [kw for kw in NI_keywords if kw in description]
    elif notif_type == 'NC':
        keywords = [kw for kw in NC_keywords if kw in description]
    else:
        keywords = []
    return ', '.join(keywords) if keywords else 'None'

def extract_location_keywords(row, desc_col, keyword_list):
    description = preprocess_keywords(row[desc_col])
    if keyword_list == clv_living_quarters_keywords:
        return 'LQ' if any(kw in description for kw in clv_living_quarters_keywords) else 'None'
    else:
        locations = [kw for kw in keyword_list if kw in description]
        return ', '.join(locations) if locations else 'None'

def create_pivot_table(df, index, columns, aggfunc='size', fill_value=0):
    """Create pivot table from dataframe"""
    df_exploded = df.assign(Keywords=df[columns].str.split(', ')).explode('Keywords')
    df_exploded = df_exploded[df_exploded['Keywords'] != 'None']
    pivot = pd.pivot_table(df_exploded, index=index, columns='Keywords', aggfunc=aggfunc, fill_value=fill_value)
    return pivot

def apply_fpso_colors(df):
    """Apply color styling to FPSO dataframe"""
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    color_map = {'GIR': '#FFA07A', 'DAL': '#ADD8E6', 'PAZ': '#D8BFD8', 'CLV': '#90EE90'}
    for fpso, color in color_map.items():
        if fpso in df.index:
            styles.loc[fpso] = f'background-color: {color}'
    return styles

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

# Sidebar file upload and FPSO selection
st.sidebar.title("Upload Notifications Dataset")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

# Add FPSO selection dropdown in the sidebar
selected_fpso = st.sidebar.selectbox("Select FPSO for Layout", ['GIR', 'DAL', 'PAZ', 'CLV'])

# NI/NC keywords (if not already in utils.py, move them there)
NI_keywords = ['WRAP', 'WELD', 'TBR', 'PACH', 'PATCH', 'OTHE', 'CLMP', 'REPL', 
               'BOND', 'BOLT', 'SUPP', 'OT', 'GASK', 'CLAMP']
NC_keywords = ['COA', 'ICOA', 'CUSP', 'WELD', 'REPL', 'CUSP1', 'CUSP2']

DB_PATH = 'notifs_data.db'
TABLE_NAME = 'notifications'

# Utility to save DataFrame to SQLite
def save_df_to_db(df, db_path=DB_PATH, table_name=TABLE_NAME):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)

# Utility to load DataFrame from SQLite
def load_df_from_db(db_path=DB_PATH, table_name=TABLE_NAME):
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql(f'SELECT * FROM {table_name}', conn)
        except Exception:
            return None

# Set Tw Cen MT font for the entire app
st.markdown(
    '''<style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stDataFrame, .stTable, .stTextInput, .stSelectbox, .stButton, .stRadio, .stSubheader, .stHeader, .stTitle, .stTabs, .stTab, .stSidebar, .stInfo, .stAlert, .stDataFrame th, .stDataFrame td {
        font-family: "Tw Cen MT", "Arial", sans-serif !important;
    }
    </style>''',
    unsafe_allow_html=True
)

# Main app logic
if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, sheet_name='Global Notifications')
        # Save to DB for persistence
        save_df_to_db(df)
        # Remove unnecessary DataFrame cast, as pd.read_excel always returns a DataFrame
        # If df is ever converted to a numpy array, ensure to convert it back to DataFrame before using .isin() or .apply()
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Define expected columns with corrected spelling
        expected_columns = {
            'Notifictn type': 'Notifictn type',  # Corrected spelling
            'Created on': 'Created on',          # Corrected spelling
            'Description': 'Description',
            'FPSO': 'FPSO'
        }
        
        # Check if all expected columns are present and map them
        missing_columns = []
        column_mapping = {}
        for expected, actual in expected_columns.items():
            if actual in df.columns:
                column_mapping[expected] = actual
            else:
                missing_columns.append(actual)
        
        if missing_columns:
            st.error(f"The following expected columns are missing: {missing_columns}")
            st.write("Please ensure your Excel file contains these columns with the exact names.")
            st.stop()
        
        # Rename columns for consistency in processing
        df = df[list(column_mapping.values())]
        df.columns = list(expected_columns.keys())
        # Ensure df is a DataFrame after slicing
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Preprocess FPSO: Keep only GIR, DAL, PAZ, CLV
        valid_fpsos = ['GIR', 'DAL', 'PAZ', 'CLV']
        df = df[df['FPSO'].isin(valid_fpsos)]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Extract NI/NC keywords
        df['Extracted_Keywords'] = df.apply(extract_ni_nc_keywords, axis=1, args=('Notifictn type', 'Description'))
        
        # Extract location keywords (modules, racks, etc.)
        df['Extracted_Modules'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_module_keywords))
        df['Extracted_Racks'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_rack_keywords))
        df['Extracted_LivingQuarters'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_living_quarters_keywords))
        df['Extracted_Flare'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_flare_keywords))
        df['Extracted_FWD'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_fwd_keywords))
        df['Extracted_HeliDeck'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_hexagons_keywords))
        
        # Extract PAZ-specific location keywords
        df['Extracted_PAZ_Modules'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_module_keywords))
        df['Extracted_PAZ_Racks'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_rack_keywords))
        df['Extracted_PAZ_LivingQuarters'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_living_quarters_keywords))
        df['Extracted_PAZ_Flare'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_flare_keywords))
        df['Extracted_PAZ_FWD'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_fwd_keywords))
        df['Extracted_PAZ_HeliDeck'] = df.apply(extract_location_keywords, axis=1, args=('Description', paz_hexagons_keywords))
        
        # Extract DAL-specific location keywords
        df['Extracted_DAL_Modules'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_module_keywords))
        df['Extracted_DAL_Racks'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_rack_keywords))
        df['Extracted_DAL_LivingQuarters'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_living_quarters_keywords))
        df['Extracted_DAL_Flare'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_flare_keywords))
        df['Extracted_DAL_FWD'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_fwd_keywords))
        df['Extracted_DAL_HeliDeck'] = df.apply(extract_location_keywords, axis=1, args=('Description', dal_hexagons_keywords))
        
        # Split dataframe into NI and NC
        df_ni = df[df['Notifictn type'] == 'NI'].copy()
        if not isinstance(df_ni, pd.DataFrame):
            df_ni = pd.DataFrame(df_ni)
        df_nc = df[df['Notifictn type'] == 'NC'].copy()
        if not isinstance(df_nc, pd.DataFrame):
            df_nc = pd.DataFrame(df_nc)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["NI Notifications", "NC Notifications", "Summary Stats", "FPSO Layout"])

        # NI Notifications Tab
        with tab1:
            st.subheader("NI Notifications Analysis")
            if not df_ni.empty:
                ni_pivot = create_pivot_table(df_ni, index='FPSO', columns='Extracted_Keywords')
                st.write("Pivot Table (Count of Keywords by FPSO):")
                styled_ni_pivot = ni_pivot.style.apply(apply_fpso_colors, axis=None)
                st.dataframe(styled_ni_pivot)
                st.write(f"Total NI Notifications: {df_ni.shape[0]}")
            else:
                st.write("No NI notifications found in the dataset.")

        # NC Notifications Tab
        with tab2:
            st.subheader("NC Notifications Analysis")
            if not df_nc.empty:
                nc_pivot = create_pivot_table(df_nc, index='FPSO', columns='Extracted_Keywords')
                st.write("Pivot Table (Count of Keywords by FPSO):")
                styled_nc_pivot = nc_pivot.style.apply(apply_fpso_colors, axis=None)
                st.dataframe(styled_nc_pivot)
                st.write(f"Total NC Notifications: {df_nc.shape[0]}")
            else:
                st.write("No NC notifications found in the dataset.")

        # NI Summary 2025 Tab
        with tab3:
            st.subheader("2025 Raised")
            # Filter for notifications in 2025
            created_on_series = pd.to_datetime(df['Created on'])
            df_2025 = df[created_on_series.dt.year == 2025].copy()
            if not df_2025.empty:
                # Add 'Month' column for monthly analysis
                df_2025['Month'] = pd.to_datetime(df_2025['Created on']).dt.strftime('%b')
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df_2025['Month'] = pd.Categorical(df_2025['Month'], categories=months_order, ordered=True)
                # Group by FPSO, Month, and Notification Type
                summary = df_2025.groupby(['FPSO', 'Month', 'Notifictn type']).size().unstack(fill_value=0)
                # Reshape the data for NI and NC notifications
                ni_summary = summary['NI'].unstack(level='Month') if 'NI' in summary else pd.DataFrame(index=pd.Index([]), columns=pd.Index(months_order))
                nc_summary = summary['NC'].unstack(level='Month') if 'NC' in summary else pd.DataFrame(index=pd.Index([]), columns=pd.Index(months_order))
                ni_summary = ni_summary.reindex(columns=pd.Index(months_order), fill_value=0) if not ni_summary.empty else pd.DataFrame(index=pd.Index([]), columns=pd.Index(months_order))
                nc_summary = nc_summary.reindex(columns=pd.Index(months_order), fill_value=0) if not nc_summary.empty else pd.DataFrame(index=pd.Index([]), columns=pd.Index(months_order))
                # Display NI Summary Table
                st.write("NI's:")
                st.dataframe(
                    ni_summary.style.set_table_styles([
                        {'selector': 'thead', 'props': [('display', 'none')]}
                    ]).set_properties(**{'text-align': 'center'})
                )
                # Display NC Summary Table
                st.write("NC's:")
                st.dataframe(
                    nc_summary.style.set_table_styles([
                        {'selector': 'thead', 'props': [('display', 'none')]}
                    ]).set_properties(**{'text-align': 'center'})
                )
                # Calculate totals
                total_ni = df_2025[df_2025['Notifictn type'] == 'NI'].shape[0]
                total_nc = df_2025[df_2025['Notifictn type'] == 'NC'].shape[0]
                st.write(f"Grand Total NI Notifications: {total_ni}")
                st.write(f"Grand Total NC Notifications: {total_nc}")
            else:
                st.write("No notifications found for 2025 in the dataset.")

        with tab4:
            st.subheader("FPSO Layout Visualization")
            notification_type = st.radio("Select Notification Type", ['NI', 'NC'])
            # Count NI or NC notifications for each location type for the selected FPSO (CLV, PAZ, DAL)
            df_selected = df[df['FPSO'] == selected_fpso].copy()
            if notification_type == 'NI':
                df_selected = df_selected[df_selected['Notifictn type'] == 'NI']
            else:  # NC
                df_selected = df_selected[df_selected['Notifictn type'] == 'NC']
            # Initialize counts for all location types
            location_counts = {
                'Modules': pd.DataFrame(index=pd.Index(clv_module_keywords), columns=['Count']).fillna(0),
                'Racks': pd.DataFrame(index=pd.Index(clv_rack_keywords), columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=pd.Index(clv_living_quarters_keywords), columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=pd.Index(clv_flare_keywords), columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=pd.Index(clv_fwd_keywords), columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=pd.Index(clv_hexagons_keywords), columns=['Count']).fillna(0)
            }
            paz_location_counts = {
                'PAZ_Modules': pd.DataFrame(index=pd.Index(paz_module_keywords), columns=['Count']).fillna(0),
                'PAZ_Racks': pd.DataFrame(index=pd.Index(paz_rack_keywords), columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=pd.Index(paz_living_quarters_keywords), columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=pd.Index(paz_flare_keywords), columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=pd.Index(paz_fwd_keywords), columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=pd.Index(paz_hexagons_keywords), columns=['Count']).fillna(0)
            }
            dal_location_counts = {
                'DAL_Modules': pd.DataFrame(index=pd.Index(dal_module_keywords), columns=['Count']).fillna(0),
                'DAL_Racks': pd.DataFrame(index=pd.Index(dal_rack_keywords), columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=pd.Index(dal_living_quarters_keywords), columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=pd.Index(dal_flare_keywords), columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=pd.Index(dal_fwd_keywords), columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=pd.Index(dal_hexagons_keywords), columns=['Count']).fillna(0)
            }
            # Count notifications for each location type and placement 
            for location_type, keywords in [
                ('Modules', clv_module_keywords),
                ('Racks', clv_rack_keywords),
                ('LivingQuarters', clv_living_quarters_keywords),
                ('Flare', clv_flare_keywords),
                ('FWD', clv_fwd_keywords),
                ('HeliDeck', clv_hexagons_keywords)
            ]:
                for keyword in keywords:
                    count = df_selected[f'Extracted_{location_type}'].str.contains(keyword, na=False).sum()
                    location_counts[location_type].loc[keyword, 'Count'] = count
            for location_type, keywords in [
                ('PAZ_Modules', paz_module_keywords),
                ('PAZ_Racks', paz_rack_keywords),
                ('LivingQuarters', paz_living_quarters_keywords),
                ('Flare', paz_flare_keywords),
                ('FWD', paz_fwd_keywords),
                ('HeliDeck', paz_hexagons_keywords)
            ]:
                for keyword in keywords:
                    if location_type == 'PAZ_Modules':
                        count = df_selected['Extracted_PAZ_Modules'].str.contains(keyword, na=False).sum()
                        paz_location_counts[location_type].loc[keyword, 'Count'] = count
                    elif location_type == 'PAZ_Racks':
                        count = df_selected['Extracted_PAZ_Racks'].str.contains(keyword, na=False).sum()
                        paz_location_counts[location_type].loc[keyword, 'Count'] = count
                    else:
                        count = df_selected[f'Extracted_{location_type}'].str.contains(keyword, na=False).sum()
                        paz_location_counts[location_type].loc[keyword, 'Count'] = count
            for location_type, keywords in [
                ('DAL_Modules', dal_module_keywords),
                ('DAL_Racks', dal_rack_keywords),
                ('LivingQuarters', dal_living_quarters_keywords),
                ('Flare', dal_flare_keywords),
                ('FWD', dal_fwd_keywords),
                ('HeliDeck', dal_hexagons_keywords)
            ]:
                for keyword in keywords:
                    if location_type == 'DAL_Modules':
                        count = df_selected['Extracted_DAL_Modules'].str.contains(keyword, na=False).sum()
                        dal_location_counts[location_type].loc[keyword, 'Count'] = count
                    elif location_type == 'DAL_Racks':
                        count = df_selected['Extracted_DAL_Racks'].str.contains(keyword, na=False).sum()
                        dal_location_counts[location_type].loc[keyword, 'Count'] = count
                    else:
                        count = df_selected[f'Extracted_{location_type}'].str.contains(keyword, na=False).sum()
                        dal_location_counts[location_type].loc[keyword, 'Count'] = count
            total_lq_count = sum(
                df_selected['Extracted_LivingQuarters'].str.contains(keyword, na=False).sum()
                for keyword in clv_living_quarters_keywords
            )
            # Draw the FPSO layout and overlay notification counts
            def draw_fpso_layout(selected_unit):
                fig, ax = plt.subplots(figsize=(13, 8))
                ax.set_xlim(0, 13.5)
                ax.set_ylim(0, 3.5)
                ax.set_aspect('equal')
                ax.grid(False)
                ax.set_facecolor('#E6F3FF')
                if selected_unit == 'CLV':
                    draw_clv(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd)
                elif selected_unit == 'PAZ':
                    draw_paz(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd)
                elif selected_unit == 'DAL':
                    draw_dal(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd)
                elif selected_unit == 'GIR':
                    draw_gir(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd)
                return fig
            fig = draw_fpso_layout(selected_fpso)
            ax = fig.gca()
            # Overlay notification counts on locations for CLV and PAZ
            if selected_fpso == 'CLV':
                # Modules
                for module, (row, col) in clv_modules.items():
                    if module in clv_module_keywords:
                        count = int(location_counts['Modules'].loc[module, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the module text for clarity >> col moves horizontally in x axis whilst row moves vertically in y axis
                            ax.text(col + 0.8, row + 0.8, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Racks
                for rack, (row, col) in clv_racks.items():
                    if rack in clv_rack_keywords:
                        count = int(location_counts['Racks'].loc[rack, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the rack text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red') # This same location should be applied to the rack modules
                
                # Living Quarters (with total count)
                for lq, (row, col) in clv_living_quarters.items():
                    if total_lq_count > 0:
                        # Position count slightly above and to the right of the LQ text
                        ax.text(col + 0.7, row + 1.4, f"{total_lq_count}", 
                                ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Flare
                for flare_loc, (row, col) in clv_flare.items():
                    if flare_loc in clv_flare_keywords:
                        count = int(location_counts['Flare'].loc[flare_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the flare text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # FWD
                for fwd_loc, (row, col) in clv_fwd.items():
                    if fwd_loc in clv_fwd_keywords:
                        count = int(location_counts['FWD'].loc[fwd_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the left of the FWD text (adjusted for rotation)
                            ax.text(col + 0.75, row + 1.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Heli-deck
                for hexagon, (row, col) in clv_hexagons.items():
                    if hexagon in clv_hexagons_keywords:
                        count = int(location_counts['HeliDeck'].loc[hexagon, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the heli-deck text
                            ax.text(col + 0.2, row + 0.2, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Total counts at the bottom (matching your image)
                total_ni = df_selected[df_selected['Notifictn type'] == 'NI'].shape[0]
                total_nc = df_selected[df_selected['Notifictn type'] == 'NC'].shape[0]
                ax.text(6, 0.25, f"NI: {total_ni}\nNC: {total_nc}", ha='center', va='center', fontsize=8, weight='bold', color='red')
            
            elif selected_fpso == 'PAZ':
                # PAZ Modules
                for module, (row, col) in paz_modules.items():
                    if module in paz_module_keywords:
                        count = int(paz_location_counts['PAZ_Modules'].loc[module, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the module text
                            ax.text(col + 0.8, row + 0.8, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # PAZ Racks
                for rack, (row, col) in paz_racks.items():
                    if rack in paz_rack_keywords:
                        count = int(paz_location_counts['PAZ_Racks'].loc[rack, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the rack text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Living Quarters (with total count)
                for lq, (row, col) in paz_living_quarters.items():
                    if total_lq_count > 0:
                        # Position count slightly above and to the right of the LQ text
                        ax.text(col + 0.7, row + 1.4, f"{total_lq_count}", 
                                ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Flare
                for flare_loc, (row, col) in paz_flare.items():
                    if flare_loc in paz_flare_keywords:
                        count = int(paz_location_counts['Flare'].loc[flare_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the flare text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # FWD
                for fwd_loc, (row, col) in paz_fwd.items():
                    if fwd_loc in paz_fwd_keywords:
                        count = int(paz_location_counts['FWD'].loc[fwd_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the left of the FWD text (adjusted for rotation)
                            ax.text(col + 0.75, row + 1.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Heli-deck
                for hexagon, (row, col) in paz_hexagons.items():
                    if hexagon in paz_hexagons_keywords:
                        count = int(paz_location_counts['HeliDeck'].loc[hexagon, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the heli-deck text
                            ax.text(col + 0.2, row + 0.2, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Total counts at the bottom
                total_ni = df_selected[df_selected['Notifictn type'] == 'NI'].shape[0]
                total_nc = df_selected[df_selected['Notifictn type'] == 'NC'].shape[0]
                ax.text(6, 0.25, f"NI: {total_ni}\nNC: {total_nc}", ha='center', va='center', fontsize=8, weight='bold', color='red')
            
            elif selected_fpso == 'DAL':
                # DAL Modules
                for module, (row, col) in dal_modules.items():
                    if module in dal_module_keywords:
                        count = int(dal_location_counts['DAL_Modules'].loc[module, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the module text
                            ax.text(col + 0.8, row + 0.8, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # DAL Racks
                for rack, (row, col) in dal_racks.items():
                    if rack in dal_rack_keywords:
                        count = int(dal_location_counts['DAL_Racks'].loc[rack, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the rack text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Living Quarters (with total count)
                for lq, (row, col) in dal_living_quarters.items():
                    if total_lq_count > 0:
                        # Position count slightly above and to the right of the LQ text
                        ax.text(col + 0.7, row + 1.4, f"{total_lq_count}", 
                                ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Flare
                for flare_loc, (row, col) in dal_flare.items():
                    if flare_loc in dal_flare_keywords:
                        count = int(dal_location_counts['Flare'].loc[flare_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the flare text
                            ax.text(col + 0.7, row + 0.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # FWD
                for fwd_loc, (row, col) in dal_fwd.items():
                    if fwd_loc in dal_fwd_keywords:
                        count = int(dal_location_counts['FWD'].loc[fwd_loc, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the left of the FWD text (adjusted for rotation)
                            ax.text(col + 0.75, row + 1.4, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Heli-deck
                for hexagon, (row, col) in dal_hexagons.items():
                    if hexagon in dal_hexagons_keywords:
                        count = int(dal_location_counts['HeliDeck'].loc[hexagon, 'Count'])
                        if count > 0:
                            # Position count slightly above and to the right of the heli-deck text
                            ax.text(col + 0.2, row + 0.2, f"{count}", 
                                    ha='center', va='center', fontsize=6, weight='bold', color='red')
                
                # Total counts at the bottom
                total_ni = df_selected[df_selected['Notifictn type'] == 'NI'].shape[0]
                total_nc = df_selected[df_selected['Notifictn type'] == 'NC'].shape[0]
                ax.text(6, 0.25, f"NI: {total_ni}\nNC: {total_nc}", ha='center', va='center', fontsize=8, weight='bold', color='red')
            
            else:
                # Display placeholder text for non-implemented FPSOs
                ax.text(6, 1.75, f"{selected_fpso} Layout\n(Implementation work in progress...)", ha='center', va='center', fontsize=16, weight='bold')
            
            plt.title(f"FPSO Visualization - {selected_fpso}", fontsize=16)
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write('Please upload an Excel file to proceed.') 
