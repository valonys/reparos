import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import matplotlib.transforms as transforms

# Sidebar file upload and FPSO selection
st.sidebar.title("Upload Notifications Dataset")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

# Add FPSO selection dropdown in the sidebar
selected_fpso = st.sidebar.selectbox("Select FPSO for Layout", ['GIR', 'DAL', 'PAZ', 'CLV'])

# Define keyword lists with grouping
NI_keywords = ['WRAP', 'WELD', 'TBR', 'PACH', 'PATCH', 'OTHE', 'CLMP', 'REPL', 
               'BOND', 'BOLT', 'SUPP', 'OT', 'GASK', 'CLAMP']
NC_keywords = ['COA', 'ICOA', 'CUSP', 'WELD', 'REPL', 'CUSP1', 'CUSP2']


# Define all location keywords based on module_layout.txt (for CLV initially)
clv_module_keywords = ['M110', 'M111', 'M112', 'M113', 'M114', 'M115', 'M116', 'H151',
                   'M120', 'M121', 'M122', 'M123', 'M124', 'M125', 'M126', 'M151']
clv_rack_keywords = ['141', '142', '143', '144', '145', '146']
clv_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
clv_flare_keywords = ['131']
clv_fwd_keywords = ['FWD']
clv_hexagons_keywords = ['HELIDECK']

# PAZ-specific keywords
paz_module_keywords = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
paz_rack_keywords = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
paz_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
paz_flare_keywords = ['FLARE']
paz_fwd_keywords = ['FWD']
paz_hexagons_keywords = ['HELIDECK']

# DAL-specific keywords
dal_module_keywords = ['P11', 'P21', 'P31', 'P41', 'P51', 'P61', 'P12', 'P22', 'P32', 'P42', 'P52', 'P62']
dal_rack_keywords = ['R11', 'R12', 'R13', 'R14', 'R15', 'R16']
dal_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
dal_flare_keywords = ['FLARE']
dal_fwd_keywords = ['FWD']
dal_hexagons_keywords = ['HELIDECK']

# Mapping for keyword grouping (including COA variations)
NI_keyword_map = {
    'TBR1': 'TBR', 'TBR2': 'TBR', 'TBR3': 'TBR', 'TBR4': 'TBR'
}
NC_keyword_map = {
    'COA1': 'COA', 'COA2': 'COA', 'COA3': 'COA', 'COA4': 'COA', 'COAT': 'COA', 'COAT1': 'COA', 'COAT2': 'COA', 'COAT3': 'COA', 'COAT4': 'COA', 'COATING': 'COA',
    'CO3': 'COA', 'C0A1': 'COA', 'C0A2': 'COA', 'C0A3': 'COA', 'C0A4': 'COA'
}

# Define CLV location dictionaries as global variables
clv_modules = {
    'M120': (0.75, 2), 'M121': (0.5, 3), 'M122': (0.5, 4), 'M123': (0.5, 5),
    'M124': (0.5, 6), 'M125': (0.5, 7), 'M126': (0.5, 8), 'M151': (0.5, 9), 'M110': (1.75, 2),
    'M111': (2, 3), 'M112': (2, 4), 'M113': (2, 5), 'M114': (2, 6),
    'M115': (2, 7), 'M116': (2, 8), 'H151': (2, 9)
}
clv_racks = {
    '141': (1.5, 3), '142': (1.5, 4), '143': (1.5, 5),
    '144': (1.5, 6), '145': (1.5, 7), '146': (1.5, 8)
}
clv_flare = {'131': (1.5, 9)}

clv_living_quarters = {'LQ': (0.5, 1)}

clv_hexagons = {'HELIDECK': (2.75, 1)}
clv_fwd = {'FWD': (0.5, 10)}

# Define PAZ location dictionaries as global variables
paz_modules = {
    'L1': (0.75, 2), 'P1': (0.5, 3), 'P2': (0.5, 4), 'P3': (0.5, 5), 'P4': (0.5, 6),
    'P5': (0.5, 7), 'P6': (0.5, 8), 'P7': (0.5, 9), 'P8': (0.5, 10), 'L2': (1.75, 2),
    'S1': (2, 3), 'S2': (2, 4), 'S3': (2, 5), 'S4': (2, 6),
    'S5': (2, 7), 'S6': (2, 8), 'S7': (2, 9), 'S8': (2, 10)
}
paz_racks = {
    'R1': (1.5, 3), 'R2': (1.5, 4), 'R3': (1.5, 5),
    'R4': (1.5, 6), 'R5': (1.5, 7), 'R6': (1.5, 8),
    'R7': (1.5, 9), 'R8': (1.5, 10)
}
paz_flare = {'FLARE': (0.5, 11)}

paz_living_quarters = {'LQ': (0.5, 1)}

paz_hexagons = {'HELIDECK': (2.75, 1)}
paz_fwd = {'FWD': (0.5, 11.75)}

# Define DAL location dictionaries as global variables
dal_modules = {
    'P11': (0.5, 2), 'P21': (0.5, 3), 'P31': (0.5, 4), 'P41': (0.5, 5),
    'P51': (0.5, 6), 'P61': (0.5, 7), 'P12': (1.75, 2), 'P22': (2, 3),
    'P32': (2, 4), 'P42': (2, 5), 'P52': (2, 6), 'P62': (2, 7)
}
dal_racks = {
    'R11': (1.5, 3), 'R12': (1.5, 4), 'R13': (1.5, 5),
    'R14': (1.5, 6), 'R15': (1.5, 7), 'R16': (1.5, 8)
}
dal_flare = {'FLARE': (1.5, 9)}

dal_living_quarters = {'LQ': (0.5, 1)}

dal_hexagons = {'HELIDECK': (2.75, 1)}
dal_fwd = {'FWD': (0.5, 10)}



# Function to preprocess and group keywords in Description
def preprocess_keywords(description):
    description = str(description).upper()
    
    # Handle all LQ variations to map to LQ
    for lq_variant in clv_living_quarters_keywords:
        if lq_variant != 'LQ':
            description = description.replace(lq_variant, 'LQ')
    
    # Handle numbers to map to corresponding module keywords (CLV)
    for module in clv_module_keywords:
        # Extract the number from the module keyword (e.g., 'M110' → '110')
        number = module[1:]  # Remove the 'M' prefix
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
    
    # Handle DAL module keywords
    for module in dal_module_keywords:
        if module in description:
            description = description.replace(module, module)
    
    # Handle DAL rack keywords
    for rack in dal_rack_keywords:
        if rack in description:
            description = description.replace(rack, rack)
    
    for original, grouped in NI_keyword_map.items():
        description = description.replace(original, grouped)
    for original, grouped in NC_keyword_map.items():
        description = description.replace(original, grouped)
    
    return description

# Function to extract NI/NC keywords based on type
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

# Function to extract location keywords (modules, racks, etc.)
def extract_location_keywords(row, desc_col, keyword_list):
    description = preprocess_keywords(row[desc_col])
    if keyword_list == clv_living_quarters_keywords:
        # For LQ, if any LQ variant is found, return 'LQ'
        return 'LQ' if any(kw in description for kw in clv_living_quarters_keywords) else 'None'
    else:
        locations = [kw for kw in keyword_list if kw in description]
        return ', '.join(locations) if locations else 'None'

# Function to create pivot table with exploded keywords
def create_pivot_table(df, index, columns, aggfunc='size', fill_value=0):
    # Explode the comma-separated keywords into individual rows
    df_exploded = df.assign(Keywords=df[columns].str.split(', ')).explode('Keywords')
    # Filter out 'None' entries
    df_exploded = df_exploded[df_exploded['Keywords'] != 'None']
    # Create pivot table
    pivot = pd.pivot_table(
        df_exploded,
        index=index,
        columns='Keywords',
        aggfunc=aggfunc,
        fill_value=fill_value
    )
    return pivot

# Function to apply custom background colors to specific FPSO rows, including index labels
def apply_fpso_colors(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    color_map = {
        'GIR': '#FFA07A',    # Orange (light salmon)
        'DAL': '#ADD8E6',    # Light Blue
        'PAZ': '#D8BFD8',    # Light Violet (thistle)
        'CLV': '#90EE90'     # Light Green
    }
    for fpso, color in color_map.items():
        if fpso in df.index:
            # Apply color to the entire row, including the index label
            styles.loc[fpso] = f'background-color: {color}'
    return styles

# Functions from module_layout.txt (updated to use global dictionaries)
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

def draw_clv(ax):
    for module, (row, col) in clv_modules.items():
        if module == 'M110':
            height, y_position, text_y = 1.25, row, row + 0.5
        elif module == 'M120':
            height, y_position, text_y = 1.25, row - 0.25, row + 0.25
        else:
            height, y_position, text_y = 1, row, row + 0.5
        add_chamfered_rectangle(ax, (col, y_position), 1, height, 0.1, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, text_y, module, ha='center', va='center', fontsize=7, weight='bold')

    for rack, (row, col) in clv_racks.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, rack, ha='center', va='center', fontsize=7, weight='bold')

    for flare_loc, (row, col) in clv_flare.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, flare_loc, ha='center', va='center', fontsize=7, weight='bold') 

    for living_quarter, (row, col) in clv_living_quarters.items():
        add_rectangle(ax, (col, row), 1, 2.5, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 1.25, living_quarter, ha='center', va='center', fontsize=7, rotation=90, weight='bold')

    for hexagon, (row, col) in clv_hexagons.items():
        add_hexagon(ax, (col, row), 0.60, edgecolor='black', facecolor='white')
        ax.text(col, row, hexagon, ha='center', va='center', fontsize=7, weight='bold')

    for fwd_loc, (row, col) in clv_fwd.items():
        add_fwd(ax, (col, row), 2.5, -1, edgecolor='black', facecolor='white')

def draw_paz(ax):
    for module, (row, col) in paz_modules.items():
        if module == 'L2':
            height, y_position, text_y = 1.25, row, row + 0.5
        elif module == 'L1':
            height, y_position, text_y = 1.25, row - 0.25, row + 0.25
        else:
            height, y_position, text_y = 1, row, row + 0.5
        add_chamfered_rectangle(ax, (col, y_position), 1, height, 0.1, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, text_y, module, ha='center', va='center', fontsize=7, weight='bold')

    for rack, (row, col) in paz_racks.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, rack, ha='center', va='center', fontsize=7, weight='bold')

    for flare_loc, (row, col) in paz_flare.items():
        add_chamfered_rectangle(ax, (col, row), 0.75, 2.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.35, row + 1.25, flare_loc, ha='center', va='center', fontsize=7, weight='bold') 

    for living_quarter, (row, col) in paz_living_quarters.items():
        add_rectangle(ax, (col, row), 1, 2.5, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 1.25, living_quarter, ha='center', va='center', fontsize=7, rotation=90, weight='bold')

    for hexagon, (row, col) in paz_hexagons.items():
        add_hexagon(ax, (col, row), 0.60, edgecolor='black', facecolor='white')
        ax.text(col, row, hexagon, ha='center', va='center', fontsize=7, weight='bold')

    for fwd_loc, (row, col) in paz_fwd.items():
        add_fwd(ax, (col, row), 2.5, -1, edgecolor='black', facecolor='white')

def draw_dal(ax):
    for module, (row, col) in dal_modules.items():
        if module == 'P11':
            height, y_position, text_y = 1.25, row, row + 0.5
        elif module == 'P12':
            height, y_position, text_y = 1.25, row - 0.25, row + 0.25
        else:
            height, y_position, text_y = 1, row, row + 0.5
        add_chamfered_rectangle(ax, (col, y_position), 1, height, 0.1, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, text_y, module, ha='center', va='center', fontsize=7, weight='bold')

    for rack, (row, col) in dal_racks.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, rack, ha='center', va='center', fontsize=7, weight='bold')

    for flare_loc, (row, col) in dal_flare.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, flare_loc, ha='center', va='center', fontsize=7, weight='bold') 

    for living_quarter, (row, col) in dal_living_quarters.items():
        add_rectangle(ax, (col, row), 1, 2.5, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 1.25, living_quarter, ha='center', va='center', fontsize=7, rotation=90, weight='bold')

    for hexagon, (row, col) in dal_hexagons.items():
        add_hexagon(ax, (col, row), 0.60, edgecolor='black', facecolor='white')
        ax.text(col, row, hexagon, ha='center', va='center', fontsize=7, weight='bold')

    for fwd_loc, (row, col) in dal_fwd.items():
        add_fwd(ax, (col, row), 2.5, -1, edgecolor='black', facecolor='white')

def draw_gir(ax):
    ax.text(6, 1.75, "GIR Layout\n(Implementation work in progress...)", ha='center', va='center', fontsize=16, weight='bold')

def draw_fpso_layout(selected_unit):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.5)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_facecolor('#E6F3FF')

    if selected_unit == 'CLV':
        draw_clv(ax)
    elif selected_unit == 'PAZ':
        draw_paz(ax)
    elif selected_unit == 'DAL':
        draw_dal(ax)
    elif selected_unit == 'GIR':
        draw_gir(ax)

    return fig

# Main app logic
if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, sheet_name='Global Notifications')
        
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
        
        # Preprocess FPSO: Keep only GIR, DAL, PAZ, CLV
        valid_fpsos = ['GIR', 'DAL', 'PAZ', 'CLV']
        df = df[df['FPSO'].isin(valid_fpsos)]
        
        # Extract NI/NC keywords
        df['Extracted_Keywords'] = df.apply(extract_ni_nc_keywords, axis=1, args=('Notifictn type', 'Description'))
        
        # Extract location keywords (modules, racks, etc.)
        df['Extracted_Modules'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_module_keywords))
        df['Extracted_Racks'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_rack_keywords))
        df['Extracted_LivingQuarters'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_living_quarters_keywords))
        df['Extracted_Flare'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_flare_keywords))
        df['Extracted_FWD'] = df.apply(extract_location_keywords, axis=1, args=('Description', clv_fwd_keywords))  # Fixed 'arguments' to 'args'
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
        df_nc = df[df['Notifictn type'] == 'NC'].copy()
        
        # Function to create pivot table with exploded keywords
        def create_pivot_table(df, index, columns, aggfunc='size', fill_value=0):
            # Explode the comma-separated keywords into individual rows
            df_exploded = df.assign(Keywords=df[columns].str.split(', ')).explode('Keywords')
            # Filter out 'None' entries
            df_exploded = df_exploded[df_exploded['Keywords'] != 'None']
            # Create pivot table
            pivot = pd.pivot_table(
                df_exploded,
                index=index,
                columns='Keywords',
                aggfunc=aggfunc,
                fill_value=fill_value
            )
            return pivot

        # Function to apply custom background colors to specific FPSO rows, including index labels
        def apply_fpso_colors(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            color_map = {
                'GIR': '#FFA07A',    # Orange (light salmon)
                'DAL': '#ADD8E6',    # Light Blue
                'PAZ': '#D8BFD8',    # Light Violet (thistle)
                'CLV': '#90EE90'     # Light Green
            }
            for fpso, color in color_map.items():
                if fpso in df.index:
                    # Apply color to the entire row, including the index label
                    styles.loc[fpso] = f'background-color: {color}'
            return styles

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
            df_2025 = df[pd.to_datetime(df['Created on']).dt.year == 2025].copy()
            
            if not df_2025.empty:
                # Add 'Month' column for monthly analysis
                df_2025['Month'] = pd.to_datetime(df_2025['Created on']).dt.strftime('%b')
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df_2025['Month'] = pd.Categorical(df_2025['Month'], categories=months_order, ordered=True)
                
                # Group by FPSO, Month, and Notification Type
                summary = df_2025.groupby(['FPSO', 'Month', 'Notifictn type']).size().unstack(fill_value=0)
                
                # Reshape the data for NI and NC notifications
                ni_summary = summary['NI'].unstack(level='Month').reindex(columns=months_order, fill_value=0)
                nc_summary = summary['NC'].unstack(level='Month').reindex(columns=months_order, fill_value=0)
                
                # Display NI Summary Table
                st.write("NI's:")
                st.dataframe(
                    ni_summary.style.set_table_styles([
                        {'selector': 'thead', 'props': [('display', 'none')]}  # Hide headers
                    ]).set_properties(**{'text-align': 'center'})  # Center-align the data
                )
                
                # Display NC Summary Table
                st.write("NC's:")
                st.dataframe(
                    nc_summary.style.set_table_styles([
                        {'selector': 'thead', 'props': [('display', 'none')]}  # Hide headers
                    ]).set_properties(**{'text-align': 'center'})  # Center-align the data
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
                'Modules': pd.DataFrame(index=clv_module_keywords, columns=['Count']).fillna(0),
                'Racks': pd.DataFrame(index=clv_rack_keywords, columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=clv_living_quarters_keywords, columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=clv_flare_keywords, columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=clv_fwd_keywords, columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=clv_hexagons_keywords, columns=['Count']).fillna(0)
            }
            
            # Initialize PAZ-specific location counts
            paz_location_counts = {
                'PAZ_Modules': pd.DataFrame(index=paz_module_keywords, columns=['Count']).fillna(0),
                'PAZ_Racks': pd.DataFrame(index=paz_rack_keywords, columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=paz_living_quarters_keywords, columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=paz_flare_keywords, columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=paz_fwd_keywords, columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=paz_hexagons_keywords, columns=['Count']).fillna(0)
            }
            
            # Initialize DAL-specific location counts
            dal_location_counts = {
                'DAL_Modules': pd.DataFrame(index=dal_module_keywords, columns=['Count']).fillna(0),
                'DAL_Racks': pd.DataFrame(index=dal_rack_keywords, columns=['Count']).fillna(0),
                'LivingQuarters': pd.DataFrame(index=dal_living_quarters_keywords, columns=['Count']).fillna(0),
                'Flare': pd.DataFrame(index=dal_flare_keywords, columns=['Count']).fillna(0),
                'FWD': pd.DataFrame(index=dal_fwd_keywords, columns=['Count']).fillna(0),
                'HeliDeck': pd.DataFrame(index=dal_hexagons_keywords, columns=['Count']).fillna(0)
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
            
            # Count PAZ-specific notifications
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

            # Count DAL-specific notifications
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

            # Calculate total LQ count by summing all LQ-related keywords
            total_lq_count = sum(
                df_selected['Extracted_LivingQuarters'].str.contains(keyword, na=False).sum()
                for keyword in clv_living_quarters_keywords
            )

            # Draw the FPSO layout and overlay notification counts
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
            st.write("Please upload an Excel file to proceed.") 
