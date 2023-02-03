#!/bin/bash
#
# This script generates a xml model out of a xacro file
# expected usage: ./xacro.sh [-n xml_output_name] [-d relpath_output_directory] ref_xacro_path
# 
# by default, it mirrors the 'models' folder structure in 'bin/models'

: ' PATH DEPENDENT FUNCTIONS '

get_bin_dir() {
    echo "${REHYB_MUJOCO_PATH}/bin"
}

get_xacro_py_path() {
    echo "${REHYB_MUJOCO_PATH}/src_py/xacro"
}

: ' GENERAL PURPOSE UTIL FUNCTIONS '

err() { echo "$*" >&2; exit 1; }

usage() {
    echo "Usage: ./xacro.sh [-n xml_output_name] [-d relpath_output_directory] ref_xacro_path";
    exit 1;
}

check_xacro_file() {
    if ! [ -f "$1" ]; then
        err "$1 does not exist or is not a file.";
    fi
    if [ "${1: -6}" != ".xacro" ]; then
        err "$1 is not a .xacro file";
    fi
}

check_xml_name() {
    if [ "${1: -4}" != ".xml" ]; then
        err "$1 is not a correct .xml name";
    fi
}

check_dir() {
    if ! [ -d "$1" ]; then
        err "$1 is not a directory.";
    fi
}

: ' SCRIPT SPECIFIC FUNCTIONS '

get_bin_model_subfolder() {
    # $1 should be the relative path to the xacro model
    # it should be in the 'models' subfolder

    # returns the absolute path to the mirror
    # subfolder in the bin

    # ex: $1==models/exo_with_patient/nesm.xacro
    # => path/to/rehyb_mujoco/bin/models/exo_with_patient
    BIN_DIR=$( get_bin_dir )
    echo "${BIN_DIR}/${1%/*}"
}

default_bin_models_folder() {
    echo "$( get_bin_dir )/models/default"
}

request_bin_folder_creation() {
    # $1 should be the candidate folder

    # returns the candidate if it is created
    # returns the default folder otherwise
    retval=""
    while true; do
        read -p "${1} does not exist. Should it be created? " yn
        case $yn in
            [Yy]* ) init_bin_folder $1; retval="${1}"; break;;
            [Nn]* ) retval="$( default_bin_models_folder )"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}

init_bin_folder() {
    # $1 should be the candidate folder path

    # create dir
    mkdir -p $1

    # add gitignore file
    GITIGNORE_PATH="${1}/.gitignore"
    echo "# Ignore everything in this directory" >> $GITIGNORE_PATH
    echo "*" >> $GITIGNORE_PATH
    echo "# Except this file" >> $GITIGNORE_PATH
    echo "!.gitignore" >> $GITIGNORE_PATH
}

use_default_directory() {
    # subroutine to be executed if no directory is specified
    # returns the output directory
    # $1 should be the relative path to the xacro model

    retval=""
    if [[ $1 == models/* ]] ;
    then
        # extract model sub folder and concatenate to bin dir
        MIRROR_DIR=$( get_bin_model_subfolder $1 )
        if [ -d $MIRROR_DIR ] ; 
        then
            # mirror folder already exists; use it
            retval="${MIRROR_DIR}"
        else
            # ask for creation of mirror folder; depending on the user choice
            # retval can be MIRROR_DIR or 'bin/models/default'
            retval=$( request_bin_folder_creation $MIRROR_DIR )
        fi
    else
        # use default directory
        retval=$( default_bin_models_folder )
    fi

    echo "$retval"
}

: ' MAIN CODE '

# get rel path to target xacro file (last argument)
# e.g. models/test/orig.xacro
REL_XACRO_PATH="${@: -1}"

# compute abs path
ABS_XACRO_PATH="$REHYB_MUJOCO_PATH/${REL_XACRO_PATH}"
check_xacro_file $ABS_XACRO_PATH

# compute output filename / directory
OUTPUT_DIRECTORY=""
OUTPUT_FILENAME=""

# check if a specific name / output directory was provided
while getopts d:n: flag
do
    case "${flag}" in
        d) OUTPUT_DIRECTORY=${OPTARG};;
        n) OUTPUT_FILENAME=${OPTARG};;
        *) usage;;
    esac
done

if [[ -z "$OUTPUT_FILENAME" ]]
then
    # no specific name provided -> use the same name as xacro file
    # get original xacro file name (e.g. orig.xacro)
    OUTPUT_FILENAME="${REL_XACRO_PATH##*/}"
    # remove xacro extension and put xml instead (e.g. orig.xml)
    OUTPUT_FILENAME="${OUTPUT_FILENAME%.*}.xml"
else
# check that the provided filename is an xml filename
    check_xml_name $OUTPUT_FILENAME
fi

if [[ -z "$OUTPUT_DIRECTORY" ]]
then
    # no target directory specified -> use 'bin/models' directory
    # check if current folder is currently replicated in the bin
    #   if yes: send to the mirror folder
    #   else: ask to initiate a new subfolder in 'bin/models'
    OUTPUT_DIRECTORY="$( use_default_directory $REL_XACRO_PATH )"
    check_dir $OUTPUT_DIRECTORY
else
    # use chosen output directory
    OUTPUT_DIRECTORY="${REHYB_MUJOCO_PATH}/${OUTPUT_DIRECTORY}"
    check_dir $OUTPUT_DIRECTORY
fi

# get output path
OUTPUT_PATH="${OUTPUT_DIRECTORY}/${OUTPUT_FILENAME}"
echo "Output path: ${OUTPUT_PATH}"

# call xacro.py
cd $( get_xacro_py_path )
python xacro.py -o $OUTPUT_PATH $ABS_XACRO_PATH