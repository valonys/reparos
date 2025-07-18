# paz.py

# PAZ-specific keywords and location dictionaries
paz_module_keywords = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
paz_rack_keywords = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
paz_living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
paz_flare_keywords = ['FLARE']
paz_fwd_keywords = ['FWD']
paz_hexagons_keywords = ['HELIDECK']

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

def draw_paz(ax, add_chamfered_rectangle, add_rectangle, add_hexagon, add_fwd):
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