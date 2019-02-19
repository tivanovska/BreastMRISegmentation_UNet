import sys 
import os 
import glob
import numpy as np
import dicom
import nibabel as nib

PATH_DICOM = './../../Mamma_Segmentation/'
PATH_DICOM_NEW = './../data/Mamma_Segmentation_corrected/'
PATH_N4ITK_OWN_MASK = './../../8.Semester(BA)/data/preprocessed_N4ITK_own_mask/'
PATH_N4ITK_OWN_MASK_NEW = './../data/preprocessed_N4ITK_own_mask/'


def correct_filename_structure(lst_files_dcm):
    if not os.path.exists(PATH_DICOM_NEW):
        os.mkdir(PATH_DICOM_NEW)
    
    # loop through all the DICOM files
    for filename_dcm in lst_files_dcm:
        # read the file
        ds = dicom.read_file(filename_dcm)
        subj_number_string = str(ds['0010', '0010'])[-5:]
        subj_number = ''
        for s in subj_number_string:
            if s.isdigit():
                subj_number += s
        #subj_number = int(subj_number)
        # store the raw image data
        instance_number_string = str(ds['0020', '0013'])[-10:]
        instance_number = ''
        for s in instance_number_string:
            if s.isdigit():
                instance_number += s
        while(len(instance_number) < 3):
            instance_number = '0' + instance_number
        file_type = filename_dcm.split('/')[-2]
        if file_type.lower().find('mammavolu') >= 0:
            file_type = 'MammaVolume'
        print(PATH_DICOM_NEW + file_type + '/subject_' + subj_number + '/' + file_type + '_subj_' + subj_number + '_slice_' + instance_number + '.dcm')
        if not os.path.exists(PATH_DICOM_NEW + file_type + '/subject_' + subj_number + '/'):
            os.makedirs(PATH_DICOM_NEW + file_type + '/subject_' + subj_number + '/')
        ds.save_as(PATH_DICOM_NEW + file_type + '/subject_' + subj_number + '/' + file_type + '_subj_' + subj_number + '_slice_' + instance_number + '.dcm')

def correct_filename_structure_nii(lst_files_nii):
    if not os.path.exists(PATH_N4ITK_OWN_MASK):
        os.mkdir(PATH_N4ITK_OWN_MASK_NEW)
    
    # loop through all the DICOM files
    for filename_nii in lst_files_nii:
        # read the file
        n4itk = nib.load(filename_nii)
        filename = filename_nii.split('/')[-1]
        subj_number = filename.split('_')[2]
        instance_number = filename.split('_')[4].split('.')[0]
        if(len(subj_number) < 2):
            subj_number = '0' + subj_number
        while(len(instance_number) < 3):
            instance_number = '0' + instance_number
        print(PATH_N4ITK_OWN_MASK_NEW + 'subject_' + subj_number + '/n4itk' + '_subj_' + subj_number + '_slice_' + instance_number + '.nii.gz')
        if not os.path.exists(PATH_N4ITK_OWN_MASK_NEW + 'subject_' + subj_number + '/'):
            os.makedirs(PATH_N4ITK_OWN_MASK_NEW + 'subject_' + subj_number + '/')
        nib.save(n4itk, PATH_N4ITK_OWN_MASK_NEW + 'subject_' + subj_number + '/n4itk' + '_subj_' + subj_number + '_slice_' + instance_number+ '.nii.gz')



if __name__ == "__main__":
    lst_files_dcm = glob.glob( PATH_DICOM + '*/*/*.dcm')
    correct_filename_structure(lst_files_dcm)
    #lst_files_nii = glob.glob( PATH_N4ITK_OWN_MASK + '*.nii.gz')
    #correct_filename_structure_nii(lst_files_nii)