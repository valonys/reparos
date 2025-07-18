# clv.py

# CLV-specific keywords and location dictionaries
clv_module_keywords = ['M110', 'M111', 'M112', 'M113', 'M114', 'M115', 'M116', 'H151',
                   'M120', 'M121', 'M122', 'M123', 'M124', 'M125', 'M126', 'M151']
clv_rack_keywords = ['141', '142', '143', '144', '145', '146']
clv_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
clv_flare_keywords = ['131']
clv_fwd_keywords = ['FWD']
clv_hexagons_keywords = ['HELIDECK']

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

def draw_clv(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd):
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