# dal.py

# DAL-specific keywords and location dictionaries
dal_module_keywords = ['P11', 'P21', 'P31', 'P41', 'P51', 'P61', 'P12', 'P22', 'P32', 'P42', 'P52', 'P62']
dal_rack_keywords = ['R11', 'R12', 'R13', 'R14', 'R15', 'R16']
dal_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
dal_flare_keywords = ['FLARE']
dal_fwd_keywords = ['FWD']
dal_hexagons_keywords = ['HELIDECK']

dal_modules = {
    'P11': (0.5, 2), 'P21': (0.5, 3), 'P31': (0.5, 4), 'P41': (0.5, 5),
    'P51': (0.5, 6), 'P61': (0.5, 7), 'P12': (2, 2), 'P22': (2, 3),
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

def draw_dal(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd):
    for module, (row, col) in dal_modules.items():
        if module == 'P11':
            height, y_position, text_y = 1, row, row + 0.5
        elif module == 'P12':
            height, y_position, text_y = 1, row, row + 0.25
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