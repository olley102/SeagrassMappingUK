"""
Sentinel-2 image batch processing

Requirements:
- ESA-SNAP with path to executable at hand
- POLYMER package loaded at PATH level
"""

import os
import logging
import subprocess
from contextlib import redirect_stdout, redirect_stderr

from polymer.main import run_atm_corr
from polymer.level1_netcdf import Level1_NETCDF
from polymer.level2_nc import Level2_NETCDF
from polymer.ancillary import Ancillary_NASA

def run_snap_gpt(graph_path, snap_executable, output_log="gpt_output.log", memory="6G"):
    """
    Run ESA SNAP subprocess with pre-defined graph file

    :param graph_path: The path to the SNAP graph file
    :param snap_executable: The path to the SNAP executable
    """

    cmd = [snap_executable, graph_path, "-c", memory]

    try:
        with open(output_log, 'w') as log_file:
            process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        print(f"SNAP processing complete. Output logged to {output_log}")
    except subprocess.CalledProcessError as e:
        print(f"Error: SNAP GPT failed. See log: {output_log}")

def create_graph_file(template, output_path="graph.xml", **kwargs):
    """
    Helper function for creating a new graph file with correctly formatted strings,
    e.g. {output_dir} can be replaced with output_dir=/path/to/outputs in kwargs
    """
    content = template.format(**kwargs)
    with open(output_path, "w") as f:
        f.write(content)
    return output_path

def process_directory_with_snap(input_dir, output_dir, snap_executable, graph_file, output_log=None):
    """
    SNAP batch processing procedure for processing an entire directory with a SNAP graph.

    :param input_dir: The directory where satellite SAFE files are stored
    :param output_dir: The directory where the outputs of this process should be stored
    :param graph_file: The path to the graph file specifying the processing steps to perform
    :param output_log: Log file for entire procedure (optional), default to output directory.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Invalid input directory for SNAP batch processing. Got {input_dir}")
    
    if output_log is None:
        output_log = os.path.join(output_dir, f'snap-log.log')

    os.makedirs(output_dir, exist_ok=True)

    with open(graph_file) as f:
        graph_template = f.read()

    safe_dirs = [x for x in os.listdir(input_dir) if x.endswith('.SAFE')]  # not absolute

    for safe_fp in safe_dirs:
        results_fp = os.path.join(output_dir,
                                  f"{safe_fp[:-5]}_snap_output")
        os.makedirs(results_fp, exist_ok=True)
        safe_abs_path = os.path.join(input_dir, safe_fp)
    
        output_file = os.path.join(results_fp, f'snap_output.nc')
        if os.path.exists(output_file):
            print(f"Output file already exists for input {safe_fp}. Skipping...")
            continue
        
        # Run SNAP process
        print("Running SNAP...")
        graph_path = create_graph_file(graph_template,
                                        input_file=safe_abs_path,
                                        output_file=output_file)
        run_snap_gpt(graph_path, snap_executable, output_log)
        print(f"Process finished. Logged to {output_log}")

def _new_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a')]
    )
    return logging.getLogger(__name__)

def run_polymer(input_file, output_file, output_log="polymer_output.log"):
    """
    Run POLYMER Atmospheric Correction process for a single NetCDF file.

    :param input_file: The input .nc file to process.
    :param output_file: The path to where the output .nc file should be.
    """

    if not input_file.endswith('.nc'):
        raise ValueError(f"Invalid input file for POLYMER. Needs .nc file extension. Got {input_file}")
    
    if not output_file.endswith('.nc'):
        raise ValueError(f"Invalid output file for POLYMER. Needs .nc file extension. Got {output_file}")
    
    try:
        level1 = Level1_NETCDF(
            filename=input_file,
            ancillary=Ancillary_NASA()
        )

        level2 = Level2_NETCDF(
            filename=output_file
        )

        logger = _new_logger(output_log)
        logger.info("Starting POLYMER Atmospheric Correction...")

        with open(output_log, 'a') as f:
            with redirect_stdout(f), redirect_stderr(f):
                run_atm_corr(
                    level1,
                    level2,
                    multiprocessing=-1
                )
        print(f"Atmospheric correction completed. Output saved to {output_file}")
        logger.info(f"Atmospheric correction completed. Output saved to {output_file}")
    except Exception as e:
        print(f"Error processing: {e}")
        logger.info("Error processing")
        logger.info(repr(e))

def process_directory_with_polymer(input_dir, output_dir, output_log=None):
    """
    POLYMER batch processing procedure for processing an entire directory with POLYMER AC.

    :param input_dir: The directory where input .nc files are stored
    :param output_dir: The directory where the outputs of this process should be stored
    :param output_log: Log file for entire procedure (optional), default to output directory.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Invalid input directory for POLYMER batch processing. Got {input_dir}")
    
    if output_log is None:
        output_log = os.path.join(output_dir, f'polymer-log.log')

    os.makedirs(output_dir, exist_ok=True)

    input_files = [x for x in os.listdir(input_dir) if x.endswith('.nc')]  # not absolute

    for input_fp in input_files:
        input_abs_path = os.path.join(input_dir, input_fp)

        output_file = os.path.join(output_dir,
                                   f"{input_fp[:-3]}_polymer_output.nc")
        if os.path.exists(output_file):
            print(f"Output file already exists for input {input_fp}. Skipping...")
            continue

        # Run POLYMER process
        print("Running POLYMER...")
        run_polymer(input_abs_path, output_file, output_log)
        print(f"Process finished. Logged to {output_log}")
    
    # TODO: need to get the tileId from the input files and add as attribute
