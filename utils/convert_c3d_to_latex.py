#!/usr/bin/env python3
"""
C3D to LaTeX Table Converter

This script converts data from a C3D binary file into a LaTeX table format.
User can control which data to extract and how many rows/columns to display.
"""

import numpy as np
import c3d
from pathlib import Path
import argparse

def read_c3d_file(file_path):
    """Read a C3D file and return the reader object."""
    try:
        # Open the C3D file for reading
        reader = c3d.Reader(open(file_path, 'rb'))
        return reader
    except Exception as e:
        print(f"Error reading C3D file: {e}")
        return None

def extract_point_data(reader, max_frames=None):
    """Extract point data from the C3D reader."""
    # Access first_frame and last_frame as properties, not methods
    if max_frames is None:
        max_frames = reader.last_frame - reader.first_frame + 1
    else:
        max_frames = min(max_frames, reader.last_frame - reader.first_frame + 1)
    
    # Get point labels
    labels = reader.point_labels
    
    # Extract 3D point data (x, y, z coordinates)
    point_data = []
    for i, points, analog in reader.read_frames(reader.first_frame, reader.first_frame + max_frames - 1):
        point_data.append(points)
    
    return np.array(point_data), labels

def determine_table_format(data_subset, precision=2):
    """
    Analyze data subset to determine appropriate siunitx table-format.
    
    Parameters:
    - data_subset: Subset of data that will be displayed in the table
    - precision: Decimal precision for output
    
    Returns:
    - table_format: String with appropriate table-format for siunitx
    """
    # Find if there are any negative values
    has_negative = np.any(data_subset < 0)
    
    # Find the maximum number of digits before decimal point
    max_val = np.max(np.abs(data_subset))
    int_digits = 1 if max_val == 0 else int(np.floor(np.log10(max_val))) + 1
    
    # Format: [sign]int_digits.precision
    sign = "-" if has_negative else ""
    return f"{sign}{int_digits}.{precision}"

def generate_latex_table(data, labels, max_rows=None, max_cols=None, caption="C3D Point Data", label="tab:c3d_data", precision=2):
    """
    Generate a LaTeX table from the data.
    
    Parameters:
    - data: 3D numpy array (frames, points, xyz)
    - labels: List of point labels
    - max_rows: Maximum number of rows to include
    - max_cols: Maximum number of point columns to include
    - caption: Table caption
    - label: Table reference label
    - precision: Decimal precision for output
    """
    if max_rows is None:
        max_rows = data.shape[0]
    else:
        max_rows = min(max_rows, data.shape[0])
    
    # Calculate number of columns (each point has x, y, z coordinates)
    if max_cols is None:
        max_cols = len(labels)
    else:
        max_cols = min(max_cols, len(labels))
    
    # Extract only the relevant subset of data for format determination
    data_subset = data[:max_rows, :min(max_cols, data.shape[1]), :]
    
    # Determine appropriate table format for only the displayed data
    table_format = determine_table_format(data_subset, precision)
    
    # Begin LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\setlength{\\tabcolsep}{5pt}\n"
    latex += "\\renewcommand{\\arraystretch}{1.2}\n"
    latex += "\\rowcolors{3}{white}{gray!10}\n"  # Start coloring from the 3rd row (after headers)
    
    # Column format with proper alignment using siunitx
    col_format = "r|"  # Frame number column right-aligned
    for _ in range(max_cols):
        col_format += f"S[table-format={table_format}]S[table-format={table_format}]S[table-format={table_format}]|"
    col_format = col_format.rstrip("|")  # Remove trailing pipe
    
    latex += "\\begin{tabular}{" + col_format + "}\n"
    latex += "\\toprule\n"
    
    # Header row with rotated text for marker names
    latex += "\\multirow{2}{*}{\\textbf{Frame}}"
    for i in range(max_cols):
        label_text = labels[i] if i < len(labels) else f"Point {i+1}"
        # For the last column, don't add the pipe separator
        column_format = "c" if i == max_cols - 1 else "c|"
        latex += f" & \\multicolumn{{3}}{{{column_format}}}{{{label_text}}}"
    latex += " \\\\\n"
    
    # Subheader for x, y, z
    latex += " "
    for _ in range(max_cols):
        latex += " & {\\textbf{x}} & {\\textbf{y}} & {\\textbf{z}}"
    latex += " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for i in range(max_rows):
        latex += f"{i+1}"
        for j in range(max_cols):
            if j < data.shape[1]:
                x, y, z = data[i, j, 0], data[i, j, 1], data[i, j, 2]
                latex += f" & {x:.{precision}f} & {y:.{precision}f} & {z:.{precision}f}"
            else:
                latex += " & {-} & {-} & {-}"  # Braces for siunitx
        latex += " \\\\\n"
        
        # Add a \midrule every 5 rows for readability if there are many rows
        if (i+1) % 5 == 0 and i < max_rows - 1:
            latex += "\\midrule\n"
    
    # End LaTeX table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    
    latex += "\\caption{" + caption + "}\n"
    latex += "\\label{" + label + "}\n"

    latex += "\\vspace{0.3cm}\n"
    latex += "\\end{table}\n"
    
    return latex

def main():
    parser = argparse.ArgumentParser(description='Convert C3D file to LaTeX table')
    parser.add_argument('c3d_file', help='Path to the C3D file')
    parser.add_argument('--output', '-o', help='Output LaTeX file', default=None)
    parser.add_argument('--rows', '-r', type=int, help='Maximum number of rows to include', default=None)
    parser.add_argument('--cols', '-c', type=int, help='Maximum number of point columns to include', default=None)
    parser.add_argument('--caption', help='Table caption', default="C3D Motion Capture Point Data")
    parser.add_argument('--label', help='Table reference label', default="tab:c3d_data")
    parser.add_argument('--units', help='Units for the data (mm, cm, m)', default="mm")
    parser.add_argument('--precision', type=int, help='Decimal precision for output', default=2)
    
    args = parser.parse_args()
    
    # Read C3D file
    reader = read_c3d_file(args.c3d_file)
    if reader is None:
        return
    
    # Extract point data
    data, labels = extract_point_data(reader, args.rows)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(
        data, labels, args.rows, args.cols, args.caption, args.label, args.precision
    )
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table written to {args.output}")
    else:
        print(latex_table)

if __name__ == "__main__":
    main()