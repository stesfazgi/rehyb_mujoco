#!/bin/bash
#
# This script simulates an XML model using MuJoCo's vanilla simulate
# Optionnally, this script calls the generation scripts of this project
# 
# To use this script, several environment variables are required; cf. README.md/Environment variables

# utils functions

err() { echo "$*" >&2; exit 1; }

print_usage() {
     printf "Usage: ./simulate.sh [-h] [-g] relative/path/to/file.xml\n\n"
     printf "Use -h flag for extensive help\n"
     exit 1
}

print_help() {
    printf "Usage: ./simulate.sh [-h] [-g] relative/path/to/file.xml\n\n"
    printf "Optional flags:\n\t-h : Prints extensive help\n\t-g : Generates included xml files\n\n"
    printf "Path to a valid xml file is required\n"
    exit 0
}

check_xml_file() {
    if [ -z "$1" ]; then
        printf "No XML file provided\n\n";
        print_usage;
    fi
    if ! [ -f "$1" ]; then
        err "$1 does not exist or is not a file.";
    fi
    if [ "${1: -4}" != ".xml" ]; then
        err "$1 is not a .xml file";
    fi
}

generate_included_files() {
    # activate python environment
    source $VIRTUALENVWRAPPER_SH
    workon $MUJOCO_PY_VENV

    # generate arm support
    echo "Generating arm support"
    ipython -c "%run src_py/xml_generation/arm_support/generate_arm_support.ipynb"

    # generate arm inertia
    echo "Generating arm inertia"
    ipython -c "%run src_py/xml_generation/arm_inertia/generate_arm_inertia.ipynb"
}


# beginning of script

if [ "$#" -eq 0 -o "$#" -gt 3 ]; then
    print_usage;
    exit 1;
fi

XML_FILE=""

while getopts 'hg' flag; do
  case "${flag}" in
    h) print_help ;;
    g) generate_included_files ;;
    ?) print_usage ;;
  esac
done

shift $(( OPTIND - 1 ))

XML_FILE="$1"
check_xml_file $XML_FILE

cd $MUJOCO_BIN_PATH
./simulate $REHYB_MUJOCO_PATH/$XML_FILE