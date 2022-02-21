# In The Name Of Allah
# Programmers: Ali Salimi and Amin Izadi
# Description: Pet Labeling Tool Application


#imports
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedStyle
from tkscrolledframe import ScrolledFrame
from PIL import Image, ImageTk
import numpy as np
import json
import pydicom
import glob
import queue
import magic
import copy
from collections import defaultdict
import math
import pylibjpeg



#global variables
VERSION = '7'



class ImageProcessing:
    def __init__(self):
        self.slices = 0
        return

    def readDicom(self, path, type):
        files = glob.glob(path + '/*')
        rows = 0
        columns = 0
        pixel_spacing = None
        slice_thickness = None
        rescale_slope = None
        rescale_intercept = None
        metadata = dict()
        patient_position = []
        counter = 0
        for file in files:
            try:
                if magic.from_file(file, mime=True) == 'application/dicom':
                    first_file = pydicom.read_file(file)
                    rows = first_file.Rows
                    columns = first_file.Columns
                    pixel_spacing = first_file.PixelSpacing
                    slice_thickness = first_file.SliceThickness
                    patient_position.append(first_file.ImagePositionPatient)
                    radio = first_file.RadiopharmaceuticalInformationSequence
                    print(radio)
                    print('ffffffffffffffffffffffffffffffffffffff')
                    
                    if type == 'PET':
                        if hasattr(first_file,'PatientID'):
                            metadata['PatientID'] = str(first_file.PatientID)
                        if hasattr(first_file,'PatientName'):
                            metadata['PatientName'] = str(first_file.PatientName)
                        if hasattr(first_file,'PatientBirthDate'):
                            metadata['PatientBirthDate'] = str(first_file.PatientBirthDate)
                        if hasattr(first_file,'PatientSex'):
                            metadata['PatientSex'] = str(first_file.PatientSex)
                        if hasattr(first_file,'InstitutionName'):
                            metadata['InstitutionName'] = str(first_file.InstitutionName)
                        if hasattr(first_file,'PatientWeight'):
                            metadata['PatientWeight'] = float(first_file.PatientWeight)
                        if hasattr(first_file,'RadiopharmaceuticalInformationSequence'):
                            radio = first_file.RadiopharmaceuticalInformationSequence
                            metadata['RadionuclideTotalDose'] = float(radio.RadionuclideTotalDose)
                            print(metadata['RadionuclideTotalDose'])
                            
                    if type == 'CT':
                        rescale_slope = first_file.RescaleSlope
                        rescale_intercept = first_file.RescaleIntercept
                    counter += 1
                    if counter == 2:
                        break
            except:
                pass

        counter = 0
        for file in files:
            if magic.from_file(file, mime=True) == 'application/dicom':
                counter += 1
        dicom = None
        if type == 'PET':
            dicom = np.empty((counter, rows, columns), dtype=np.uint32)
            for file in files:
                try:
                    if magic.from_file(file, mime=True) == 'application/dicom':
                        slice_ = pydicom.dcmread(file)
                        rescale_slope = slice_.RescaleSlope
                        rescale_intercept = slice_.RescaleIntercept
                        dicom[slice_.InstanceNumber - 1] = slice_.pixel_array * rescale_slope + rescale_intercept
                except:
                    pass
        elif type == 'CT':
            dicom = np.empty((counter, rows, columns), dtype=np.uint16)
            for file in files:
                try:
                    if magic.from_file(file, mime=True) == 'application/dicom':
                        slice_ = pydicom.dcmread(file)
                        dicom[slice_.InstanceNumber - 1] = slice_.pixel_array
                except:
                    pass

        patient_position = np.array(patient_position)
        diff = patient_position[0] - patient_position[1]
        if diff[2] > 0:
            pass
        else:
            dicom = np.flip(dicom, 0)
        dicom_max = np.amax(dicom)
        dicom_min = np.amin(dicom)
        dicom_size = dicom.shape
        if type == 'PET':
            return dicom, (dicom_min, dicom_max), tuple(dicom_size), pixel_spacing[0], slice_thickness, metadata
        else:
            return dicom, (dicom_min, dicom_max), tuple(dicom_size), pixel_spacing[0], slice_thickness, rescale_slope, rescale_intercept

    def binarization(self, volume, volume_bounds, threshold):
        threshold = threshold * (volume_bounds[1] - volume_bounds[0]) + volume_bounds[0]
        barray = np.array((volume > threshold) * 1, dtype=np.int8)
        return barray

    def RG(self, volume, coordinate, neighbor_type):
        #initialization
        volume_temp = copy.copy(volume)
        x = coordinate[0]
        y = coordinate[1]
        z = coordinate[2]
        value = volume_temp[x][y][z]
        segment = set()
        q = queue.Queue()
        #doing
        volume_temp[x][y][z] = -1
        segment.add(coordinate)
        q.put(coordinate)
        while not q.empty():
            point = q.get()
            self.G(point, volume_temp, segment, q, value, neighbor_type)
        #ending
        return segment

    def G(self, point, volume, segment, q, value, neighbor_type):
        #initialization
        x = point[0]
        y = point[1]
        z = point[2]
        neighbors = self.getNeighbors(x, y, z, neighbor_type)
        #doing
        for n in neighbors:
            nx = n[0]
            ny = n[1]
            nz = n[2]
            try:
                if volume[nx][ny][nz] == value:
                    volume[nx][ny][nz] = -1
                    segment.add(n)
                    q.put(n)
            except:
                continue
        #ending
        return

    def getNeighbors(self, x, y, z, neighbor_type):
        if neighbor_type == 'max':
            neighbors = [
                (x - 1, y - 1, z - 1),
                (x - 1, y - 1, z),
                (x - 1, y - 1, z + 1),
                (x - 1, y, z - 1),
                (x - 1, y, z),
                (x - 1, y, z + 1),
                (x - 1, y + 1, z - 1),
                (x - 1, y + 1, z),
                (x - 1, y + 1, z + 1),
                (x, y - 1, z - 1),
                (x, y - 1, z),
                (x, y - 1, z + 1),
                (x, y, z - 1),
                (x, y, z + 1),
                (x, y + 1, z - 1),
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y - 1, z - 1),
                (x + 1, y - 1, z),
                (x + 1, y - 1, z + 1),
                (x + 1, y, z - 1),
                (x + 1, y, z),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z - 1),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1)
            ]
        elif neighbor_type == 'med':
            neighbors = [
                (x - 1, y - 1, z),
                (x - 1, y, z - 1),
                (x - 1, y, z),
                (x - 1, y, z + 1),
                (x - 1, y + 1, z),
                (x, y - 1, z - 1),
                (x, y - 1, z),
                (x, y - 1, z + 1),
                (x, y, z - 1),
                (x, y, z + 1),
                (x, y + 1, z - 1),
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y - 1, z),
                (x + 1, y, z - 1),
                (x + 1, y, z),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z)
            ]
        elif neighbor_type == 'min':
            neighbors = [
                (x - 1, y, z),
                (x, y - 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
                (x, y + 1, z),
                (x + 1, y, z)
            ]
        return neighbors

    def RG2D(self, slice, coordinate, neighbor_type):
        #initialization
        slice_temp = copy.copy(slice)
        x = coordinate[0]
        y = coordinate[1]
        value = slice_temp[x][y]
        segment = set()
        q = queue.Queue()
        #doing
        slice_temp[x][y] = -1
        segment.add(coordinate)
        q.put(coordinate)
        while not q.empty():
            point = q.get()
            self.G2D(point, slice_temp, segment, q, value, neighbor_type)
        #ending
        return segment

    def G2D(self, point, slice, segment, q, value, neighbor_type):
        #initialization
        x = point[0]
        y = point[1]
        neighbors = self.getNeighbors2D(x, y, neighbor_type)
        #doing
        for n in neighbors:
            nx = n[0]
            ny = n[1]
            try:
                if slice[nx][ny] == value:
                    slice[nx][ny] = -1
                    segment.add(n)
                    q.put(n)
            except:
                continue
        #ending
        return

    def getNeighbors2D(self, x, y, neighbor_type):
        if neighbor_type == 'max':
            neighbors = [
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
            ]
        elif neighbor_type == 'min':
            neighbors = [
                (x - 1, y),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y),
            ]
        return neighbors


class Dicom:
    def __init__(self):
        self.pet_path = None
        self.ct_path = None
        self.labels_path = None
        self.save_path = None
        #
        self.pet_volume = None
        self.ct_volume = None
        self.binary_volume = None
        self.threshold_volume = None
        self.label_volume = None
        #
        self.current_view = None
        self.current_slice_number = None
        self.current_threshold = None
        #
        self.pet_bounds = None
        self.pet_size = None
        self.ct_bounds = None
        self.ct_size = None
        self.pet_pixel_spacing = None
        self.ct_pixel_spacing = None
        self.pet_slice_thickness = None
        self.ct_slice_thickness = None
        self.metadata = None
        #
        self.current_ct_window = None
        self.rescale_slope = None
        self.rescale_intercept = None
        #
        self.ip = None
        self.current_state = None
        self.log = None
        #
        return

    def getSlices(self, view=None, slice_number=None, threshold=None, ct_window=None):
        #initialization
        if view == None:
            view = self.current_view
        if slice_number == None:
            slice_number = self.current_slice_number
        if threshold == None:
            threshold = self.current_threshold
        if ct_window == None:
            ct_window = self.current_ct_window
        else:
            self.current_ct_window = ct_window

        #change view and threshold
        if self.current_view == view:
            if self.current_threshold == threshold:
                pass
            else:
                self.binary_volume = self.ip.binarization(self.pet_volume, self.pet_bounds, threshold)
                self.current_threshold = threshold
        else:
            self.pet_volume = self.rotation(self.pet_volume, view)
            self.ct_volume = self.rotation(self.ct_volume, view)
            self.threshold_volume = self.rotation(self.threshold_volume, view)
            self.label_volume = self.rotation(self.label_volume, view)
            if self.current_threshold == threshold:
                self.binary_volume = self.rotation(self.binary_volume, view)
            else:
                self.binary_volume = self.ip.binarization(self.pet_volume, self.pet_bounds, threshold)
                self.current_threshold = threshold
            self.current_view = view

        #validate slice number
        if slice_number == 'middle':
            slice_number = int(self.pet_volume.shape[0]/2)
        elif slice_number < 0:
            slice_number = 0
        elif view == 'axial' and slice_number >= self.pet_size[0]:
            slice_number = self.pet_size[0] - 1
        elif view == 'coronal' and slice_number >= self.pet_size[1]:
            slice_number = self.pet_size[1] - 1
        elif view == 'sagital' and slice_number >= self.pet_size[2]:
            slice_number = self.pet_size[2] - 1
        self.current_slice_number = slice_number

        #pet slice
        pet_slice = self.pet_volume[slice_number]
        pet_slice_max = np.max(pet_slice)
        pet_slice_min = np.min(pet_slice)
        pet_slice = (pet_slice - pet_slice_min) / (pet_slice_max - pet_slice_min)

        #ct slice
        ct_slice = None
        if view == 'axial':
            ct_actual_size = round(self.ct_size[0] * self.ct_slice_thickness / self.pet_slice_thickness)
            m_l = int((self.pet_size[0] - ct_actual_size)/2)
            m_r = self.pet_size[0] - ct_actual_size - m_l
            if slice_number < m_l or slice_number >= self.pet_size[0]-m_r:
                ct_slice = np.zeros((self.ct_size[1], self.ct_size[2]))
            else:
                norm_coord = (slice_number - m_l)/ct_actual_size
                actual_coord = round(norm_coord * self.ct_size[0])
                ct_slice = self.ct_volume[actual_coord]
        elif view == 'coronal' or view == 'sagital':
            ct_actual_size = round(self.ct_size[1] * self.ct_pixel_spacing / self.pet_pixel_spacing)
            m_l = int((self.pet_size[1] - ct_actual_size)/2)
            m_r = (self.pet_size[1] - ct_actual_size) - m_l
            if slice_number < m_l or slice_number >= self.pet_size[1]-m_r:
                ct_slice = np.zeros((self.ct_size[0], self.ct_size[1]))
            else:
                norm_coord = (slice_number - m_l)/ct_actual_size
                actual_coord = round(norm_coord * self.ct_size[1])
                ct_slice = self.ct_volume[actual_coord]
        if ct_window[0]:
            ct_window_min = (ct_window[1] - self.rescale_intercept) / self.rescale_slope
            ct_window_max = (ct_window[2] - self.rescale_intercept) / self.rescale_slope
            ct_slice = np.where(ct_slice < ct_window_min, ct_window_min, ct_slice)
            ct_slice = np.where(ct_slice > ct_window_max, ct_window_max, ct_slice)
        ct_slice_max = np.max(ct_slice)
        ct_slice_min = np.min(ct_slice)
        ct_slice = (ct_slice - ct_slice_min) / (ct_slice_max - ct_slice_min)
        #binary slice
        binary_slice = self.binary_volume[slice_number]

        #anomality slice
        anomality_slice =  self.label_volume[slice_number]

        #result
        result = dict()
        if view == 'axial':
            result['pet'] = {
                'slice': pet_slice,
                'x': self.pet_pixel_spacing,
                'y': self.pet_pixel_spacing
            }
            result['ct'] = {
                'slice': ct_slice,
                'x': self.ct_pixel_spacing,
                'y': self.ct_pixel_spacing
            }
            result['binary'] = {
                'slice': binary_slice,
                'x': self.pet_pixel_spacing,
                'y': self.pet_pixel_spacing
            }
            result['anomality'] = {
                'slice': anomality_slice,
                'x': self.pet_pixel_spacing,
                'y': self.pet_pixel_spacing
            }
        elif view == 'coronal' or view == 'sagital':
            result['pet'] = {
                'slice': pet_slice,
                'x': self.pet_slice_thickness,
                'y': self.pet_pixel_spacing
            }
            result['ct'] = {
                'slice': ct_slice,
                'x': self.ct_slice_thickness,
                'y': self.ct_pixel_spacing
            }
            result['binary'] = {
                'slice': binary_slice,
                'x': self.pet_slice_thickness,
                'y': self.pet_pixel_spacing
            }
            result['anomality'] = {
                'slice': anomality_slice,
                'x': self.pet_slice_thickness,
                'y': self.pet_pixel_spacing
            }
        return result

    def label(self, coordinate, label, neighbor_type):
        changes = None
        message = ''
        x = coordinate[0]
        y = coordinate[1]
        z = coordinate[2]
        binary_value = self.binary_volume[x][y][z]
        label_value = self.label_volume[x][y][z]
        condition = (binary_value, label_value, label)

        if condition == (0, 0, 'suspect'):
            message = 'no effect.'
        elif condition == (0, 0, 'tumoral'):
            message = 'no effect.'
        elif condition == (0, 0.5, 'suspect'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, 'max')
            for sv in segment_voxels:
                if self.binary_volume[sv[0]][sv[1]][sv[2]] == 0:
                    changes.append(['label_volume', sv, 0.5, 0])
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], -1])
                    self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
                else:
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], self.current_threshold])
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'suspect segment is modified.'
        elif condition == (0, 0.5, 'tumoral'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, 'max')
            for sv in segment_voxels:
                if self.binary_volume[sv[0]][sv[1]][sv[2]] == 0:
                    changes.append(['label_volume', sv, 0.5, 0])
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], -1])
                    self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
                else:
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], self.current_threshold])
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'suspect segment is modified.'
        elif condition == (0, 1, 'suspect'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, 'max')
            for sv in segment_voxels:
                if self.binary_volume[sv[0]][sv[1]][sv[2]] == 0:
                    changes.append(['label_volume', sv, 1, 0])
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], -1])
                    self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
                else:
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], self.current_threshold])
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'tumoral segment is modified.'
        elif condition == (0, 1, 'tumoral'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, 'max')
            for sv in segment_voxels:
                if self.binary_volume[sv[0]][sv[1]][sv[2]] == 0:
                    changes.append(['label_volume', sv, 1, 0])
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], -1])
                    self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
                else:
                    changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]], self.current_threshold])
                    self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'tumoral segment is modified.'
        elif condition == (1, 0, 'suspect'):
            changes = list()
            segment_voxels = self.ip.RG(self.binary_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 0, 0.5])
                changes.append(['threshold_volume', sv, -1, self.current_threshold])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 0.5
                self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'tumoral segment is labeled.'
        elif condition == (1, 0, 'tumoral'):
            changes = list()
            segment_voxels = self.ip.RG(self.binary_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 0, 1])
                changes.append(['threshold_volume', sv, -1, self.current_threshold])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 1
                self.threshold_volume[sv[0]][sv[1]][sv[2]] = self.current_threshold
            message = 'tumoral segment is labeled.'
        elif condition == (1, 0.5, 'suspect'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 0.5, 0])
                changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]] , -1])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
            message = 'suspect segment is deleted.'
        elif condition == (1, 0.5, 'tumoral'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 0.5, 1])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 1
            message = 'suspect segment is changed to anomal.'
        elif condition == (1, 1, 'suspect'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 1, 0.5])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 0.5
            message = 'tumoral segment is changed to suspect.'
        elif condition == (1, 1, 'tumoral'):
            changes = list()
            segment_voxels = self.ip.RG(self.label_volume, coordinate, neighbor_type)
            for sv in segment_voxels:
                changes.append(['label_volume', sv, 1, 0])
                changes.append(['threshold_volume', sv, self.threshold_volume[sv[0]][sv[1]][sv[2]] , -1])
                self.label_volume[sv[0]][sv[1]][sv[2]] = 0
                self.threshold_volume[sv[0]][sv[1]][sv[2]] = -1
            message = 'tumoral segment is deleted.'

        if changes:
            if self.current_state >= 3 :
                self.log = self.log[1:]
                self.log.append(changes)
            else:
                self.log = self.log[:self.current_state]
                self.log.append(changes)
                self.current_state += 1

        return message

    def label2D(self, coordinate, label, neighbor_type):
        changes = None
        message = ''
        x = coordinate[0]
        y = coordinate[1]
        binary_value = self.binary_volume[self.current_slice_number][x][y]
        label_value = self.label_volume[self.current_slice_number][x][y]
        condition = (binary_value, label_value, label)

        if condition == (0, 0, 'suspect'):
            message = 'no effect.'
        elif condition == (0, 0, 'tumoral'):
            message = 'no effect.'
        elif condition == (0, 0.5, 'suspect'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, 'max')
            for sp in segment_pixels:
                if self.binary_volume[self.current_slice_number][sp[0]][sp[1]] == 0:
                    changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0.5, 0])
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], -1])
                    self.label_volume[self.current_slice_number][sp[0]][sp[1]]  = 0
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
                else:
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] , self.current_threshold])
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]]  = self.current_threshold
            message = 'suspect segment is modified.'
        elif condition == (0, 0.5, 'tumoral'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, 'max')
            for sp in segment_pixels:
                if self.binary_volume[self.current_slice_number][sp[0]][sp[1]] == 0:
                    changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0.5, 0])
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], -1])
                    self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
                else:
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], self.current_threshold])
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = self.current_threshold
            message = 'suspect segment is modified.'
        elif condition == (0, 1, 'suspect'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, 'max')
            for sp in segment_pixels:
                if self.binary_volume[self.current_slice_number][sp[0]][sp[1]] == 0:
                    changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 1, 0])
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], -1])
                    self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
                else:
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], self.current_threshold])
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = self.current_threshold
            message = 'tumoral segment is modified.'
        elif condition == (0, 1, 'tumoral'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, 'max')
            for sp in segment_pixels:
                if self.binary_volume[self.current_slice_number][sp[0]][sp[1]] == 0:
                    changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 1, 0])
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], -1])
                    self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
                else:
                    changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]], self.current_threshold])
                    self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = self.current_threshold
            message = 'tumoral segment is modified.'
        elif condition == (1, 0, 'suspect'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.binary_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0, 0.5])
                changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), -1, self.current_threshold])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0.5
                self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = self.current_threshold
            message = 'tumoral segment is labeled.'
        elif condition == (1, 0, 'tumoral'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.binary_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0, 1])
                changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), -1, self.current_threshold])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 1
                self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = self.current_threshold
            message = 'tumoral segment is labeled.'
        elif condition == (1, 0.5, 'suspect'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0.5, 0])
                changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] , -1])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0
                self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
            message = 'suspect segment is deleted.'
        elif condition == (1, 0.5, 'tumoral'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 0.5, 1])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 1
            message = 'suspect segment is changed to anomal.'
        elif condition == (1, 1, 'suspect'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 1, 0.5])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0.5
            message = 'tumoral segment is changed to suspect.'
        elif condition == (1, 1, 'tumoral'):
            changes = list()
            segment_pixels = self.ip.RG2D(self.label_volume[self.current_slice_number], coordinate, neighbor_type)
            for sp in segment_pixels:
                changes.append(['label_volume', (self.current_slice_number,sp[0],sp[1]), 1, 0])
                changes.append(['threshold_volume', (self.current_slice_number,sp[0],sp[1]), self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] , -1])
                self.label_volume[self.current_slice_number][sp[0]][sp[1]] = 0
                self.threshold_volume[self.current_slice_number][sp[0]][sp[1]] = -1
            message = 'tumoral segment is deleted.'

        if changes:
            if self.current_state >= 3 :
                self.log = self.log[1:]
                self.log.append(changes)
            else:
                self.log = self.log[:self.current_state]
                self.log.append(changes)
                self.current_state += 1

        return message

    def undoRedo(self, action_type):
        is_done = False
        if action_type == 'undo' and self.current_state != 0:
            for log in self.log[self.current_state-1]:
                if log[0] == 'label_volume':
                    self.label_volume[log[1][0]][log[1][1]][log[1][2]] = log[2]
                elif log[0] == 'threshold_volume':
                    self.threshold_volume[log[1][0]][log[1][1]][log[1][2]] = log[2]
            self.current_state -= 1
            is_done = True
        elif action_type == 'redo' and self.current_state < len(self.log):
            for log in self.log[self.current_state]:
                if log[0] == 'label_volume':
                    self.label_volume[log[1][0]][log[1][1]][log[1][2]] = log[3]
                elif log[0] == 'threshold_volume':
                    self.threshold_volume[log[1][0]][log[1][1]][log[1][2]] = log[3]
            self.current_state += 1
            is_done = True

        message = str()
        if action_type == 'undo' and is_done:
            message = 'undo is done.'
        if action_type == 'undo' and not is_done:
            message = 'unable to undo.'
        if action_type == 'redo' and is_done:
            message = 'redo is done.'
        if action_type == 'redo' and not is_done:
            message = 'unable to redo.'
        return message

    def roi(self, coordinate, radius):
        uptakes = np.array([])
        radius_ps = math.ceil(radius/self.pet_pixel_spacing)
        radius_st = math.ceil(radius/self.pet_slice_thickness)
        volum_size = self.pet_volume.shape
        if self.current_view == 'axial':
            for z in range(coordinate[0] - radius_st, coordinate[0] + radius_st + 1):
                for x in range(coordinate[1] - radius_ps, coordinate[1] + radius_ps + 1):
                    for y in range(coordinate[2] - radius_ps, coordinate[2] + radius_ps + 1):
                        dist = math.sqrt(
                            ((coordinate[0] * self.pet_slice_thickness - z * self.pet_slice_thickness) ** 2)  +
                            ((coordinate[1] * self.pet_pixel_spacing - x * self.pet_pixel_spacing) ** 2) +
                            ((coordinate[2] * self.pet_pixel_spacing - y * self.pet_pixel_spacing) ** 2)
                            )
                        dim = np.array([z,x,y])
                        cond1 = dim < volum_size
                        cond2 = dim >= np.array([0,0,0])
                        if radius >= dist and all(cond1) and all(cond2):
                            uptakes = np.append(uptakes,self.pet_volume[z][x][y])
        elif self.current_view == 'coronal':
            for z in range(coordinate[0] - radius_ps, coordinate[0] + radius_ps + 1):
                for x in range(coordinate[1] - radius_st, coordinate[1] + radius_st + 1):
                    for y in range(coordinate[2] - radius_ps, coordinate[2] + radius_ps + 1):
                        dist = math.sqrt(
                            ((coordinate[0] * self.pet_pixel_spacing - z * self.pet_pixel_spacing) ** 2)  +
                            ((coordinate[1] * self.pet_slice_thickness - x * self.pet_slice_thickness) ** 2) +
                            ((coordinate[2] * self.pet_pixel_spacing - y * self.pet_pixel_spacing) ** 2)
                            )
                        dim = np.array([z,x,y])
                        cond1 = dim < volum_size
                        cond2 = dim >= np.array([0,0,0])
                        if radius >= dist and all(cond1) and all(cond2):
                            uptakes = np.append(uptakes,self.pet_volume[z][x][y])
        elif self.current_view == 'sagital':
            for z in range(coordinate[0] - radius_ps, coordinate[0] + radius_ps + 1):
                for x in range(coordinate[1] - radius_st, coordinate[1] + radius_st + 1):
                    for y in range(coordinate[2] - radius_ps, coordinate[2] + radius_ps + 1):
                        dist = math.sqrt(
                            ((coordinate[0] * self.pet_pixel_spacing - z * self.pet_pixel_spacing) ** 2)  +
                            ((coordinate[1] * self.pet_slice_thickness - x * self.pet_slice_thickness) ** 2) +
                            ((coordinate[2] * self.pet_pixel_spacing - y * self.pet_pixel_spacing) ** 2)
                            )
                        dim = np.array([z,x,y])
                        cond1 = dim < volum_size
                        cond2 = dim >= np.array([0,0,0])
                        if radius >= dist and all(cond1) and all(cond2):
                            uptakes = np.append(uptakes,self.pet_volume[z][x][y])

        uptake_max = round(np.max(uptakes), 2)
        uptake_min = round(np.min(uptakes), 2)
        uptake_mean = round(np.mean(uptakes), 2)
        suv_max = round((uptake_max*self.metadata['PatientWeight'])*1000/self.metadata['RadionuclideTotalDose'], 4)
        suv_min = round((uptake_min*self.metadata['PatientWeight'])*1000/self.metadata['RadionuclideTotalDose'], 4)
        suv_mean = round((uptake_mean*self.metadata['PatientWeight'])*1000/self.metadata['RadionuclideTotalDose'], 4)
        uptake_mean_percent = round(((uptake_mean - self.pet_bounds[0])/(self.pet_bounds[1] - self.pet_bounds[0]))*100)

        message = 'uptake max: {}\nuptake min: {}\nuptake mean: {}\nsuv max: {}\nsuv min: {}\nsuv mean: {}\nuptake mean percent: {}\n'.format(
        uptake_max, uptake_min, uptake_mean, suv_max, suv_min, suv_mean, uptake_mean_percent)

        return (uptake_mean_percent, message)

    def save(self, save_path):
        self.save_path = save_path

        #save label volume
        label_volume = self.rotation(self.label_volume, 'axial')
        indices = np.where(label_volume > 0)
        values = label_volume[indices]
        indices = np.array(indices).transpose()
        label_json = defaultdict(list)
        for i, j in zip(values, indices):
            label_json[str(i)].append(j.tolist())

        #save threshold volume
        threshold_volume = self.rotation(self.threshold_volume, 'axial')
        indices = np.where(threshold_volume > -1)
        values = threshold_volume[indices]
        indices = np.array(indices).transpose()
        threshold_json = defaultdict(list)
        for i, j in zip(values, indices):
            threshold_json[str(i)].append(j.tolist())

        results = {'version': VERSION, 'labels':label_json, 'thresholds':threshold_json, 'metadata':self.metadata}
        with open(self.save_path + '/label.json', 'w') as f:
            json.dump(results, f,ensure_ascii=False)

        return

    def open(self, pet_path, ct_path, labels_path):
        self.ip = ImageProcessing()

        del self.pet_volume
        del self.ct_volume
        del self.threshold_volume
        del self.label_volume
        del self.binary_volume

        #PET opening
        self.pet_path = pet_path
        (self.pet_volume, self.pet_bounds, self.pet_size,
        self.pet_pixel_spacing, self.pet_slice_thickness,
        self.metadata) = self.ip.readDicom(self.pet_path, 'PET')

        #CT opening
        self.ct_path = ct_path
        (self.ct_volume, self.ct_bounds, self.ct_size,
        self.ct_pixel_spacing, self.ct_slice_thickness, self.rescale_slope, self.rescale_intercept) = self.ip.readDicom(self.ct_path, 'CT')
        self.ct_slice_thickness = (self.pet_size[0]*self.pet_slice_thickness)/self.ct_size[0]

        #labels opening
        self.labels_path = labels_path
        self.threshold_volume = np.zeros(self.pet_size,dtype='float')
        self.threshold_volume = self.threshold_volume - 1
        self.label_volume = np.zeros(self.pet_size,dtype='float')
        
        try:
            with open(self.labels_path + '/label.json', 'r') as f:
                temp = json.load(f)
                labels = temp['labels']
                for key in labels:
                    for v in labels[key]:
                        self.label_volume[v[0],v[1],v[2]] = float(key)
                thresholds = temp['thresholds']
                for key in thresholds:
                    for v in thresholds[key]:
                        self.threshold_volume[v[0],v[1],v[2]] = float(key)
        except:
            pass
        self.current_view = 'axial'
        self.current_slice_number = 0
        self.current_threshold = 0
        self.binary_volume = self.ip.binarization(self.pet_volume, self.pet_bounds, self.current_threshold)
        self.current_ct_window = (False,0,0)
        self.current_state = 0
        self.log = []
        
        return

    #private
    def rotation(self, volume, view):
        if self.current_view == 'axial':
            if view == 'coronal':
                result = np.rot90(volume, 1, (0, 1))
            elif view == 'sagital':
                result = np.rot90(np.rot90(volume, 1, (2, 0)), 1, (1, 2))
            else:
                result = volume
        elif self.current_view == 'coronal':
            if view == 'axial':
                result = np.rot90(volume,1,(1,0))
            elif view == 'sagital':
                result = np.rot90(volume,1,(2,0))
            else:
                result = volume
        elif self.current_view == 'sagital':
            if view == 'axial':
                result = np.rot90(np.rot90(volume,1,(2,1)),1,(0,2))
            elif view == 'coronal':
                result = np.rot90(volume,1,(0,2))
            else:
                result = volume
        return result


class PltApp:
    def __init__(self, master):
        self.dicom = Dicom()
        self.paths = {
            'pet': None,
            'ct': None,
            'label': None
        }
        self.entities = dict()
        self.fire_palette = list()
        self.mask_palette = [0, 0, 0]*128 + [255, 0, 255]*128
        self.anomality_palette = [0, 0, 0]*26 + [0, 255, 0]*51 + [255, 255, 0]*51 + [255, 127, 0]*51 + [255, 255, 0]*51 + [255, 0, 0]*26
        self.image_shape = None
        self.roi_image = Image.open('roi.png')

        self.slice_number = IntVar()
        self.image_view = StringVar()
        #
        self.pet_show = BooleanVar()
        self.pet_opacity = IntVar()
        self.pet_color = StringVar()
        #
        self.ct_show = BooleanVar()
        self.ct_opacity = IntVar()
        self.ct_color = StringVar()
        #
        self.binary_show = BooleanVar()
        self.binary_color = StringVar()
        self.binary_opacity = IntVar()
        #
        self.ct_window = BooleanVar()
        self.ct_window_min = IntVar()
        self.ct_window_max = IntVar()
        self.binary_threshold = DoubleVar()
        self.label_in_use = StringVar()
        self.tool_in_hand = StringVar()
        self._2d_labeler_growth = StringVar()
        self._3d_labeler_growth = StringVar()
        self.roi_show = False
        self.roi_coordinate = None
        self.roi_radius = IntVar()
        self.roi_ump = IntVar()
        #
        self.zoom_level = 0
        self.state = None

        #fire palette fill
        self.fire_palette.extend([0, 0, 0])
        for i in range(2, 255, 3):
            self.fire_palette.extend([i, 0, 0])
        for i in range(2, 255, 3):
            self.fire_palette.extend([255, i, 0])
        for i in range(2, 255, 3):
            self.fire_palette.extend([255, 255, i])

        #master
        self.master = master
        self.master.title('PLT Application')
        self.master.geometry('900x480+200+200')
        self.master.state('zoomed')
        self.style = ThemedStyle(self.master)
        self.style.theme_use('equilux')
        self.master.protocol("WM_DELETE_WINDOW", self.exit)

        #panedwindow
        self.panedwindow = ttk.Panedwindow(self.master, orient=HORIZONTAL)
        self.frame_desk = ttk.Frame(self.master, relief=GROOVE)
        self.frame_toolbox = ttk.Frame(self.master)
        #
        self.panedwindow.pack(fill=BOTH, expand=True)
        self.panedwindow.add(self.frame_desk, weight=10)
        self.panedwindow.add(self.frame_toolbox)

        #frame_toolbox
        self.frame_navbar = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_fov = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_ct_window = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_threshold = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_label_picker = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_2d_labeler = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_3d_labeler = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_roi = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_undo_redo = ttk.Frame(self.frame_toolbox, relief=RAISED)
        self.frame_info = ttk.Frame(self.frame_toolbox, relief=RAISED)
        #
        self.frame_navbar.pack(fill=X, padx=3, pady=3)
        self.frame_fov.pack(fill=X, padx=3, pady=3)
        self.frame_ct_window.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_threshold.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_label_picker.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_2d_labeler.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_3d_labeler.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_roi.pack(fill=X, padx=3, pady=3, ipadx=3, ipady=3)
        self.frame_undo_redo.pack(fill=X, padx=3, pady=3)
        self.frame_info.pack(fill=BOTH, padx=3, pady=3)

        #frame_navbar
        self.button_open = ttk.Button(self.frame_navbar, text='Open', command=self.open)
        self.button_save = ttk.Button(self.frame_navbar, text='Save', command=self.save)
        self.button_exit = ttk.Button(self.frame_navbar, text='Exit', command=self.exit)
        #
        self.button_open.pack(side=LEFT, padx=3, pady=3)
        self.button_save.pack(side=LEFT, padx=3, pady=3)
        self.button_exit.pack(side=LEFT, padx=3, pady=3)

        #frame_fov
        self.frame_slice = ttk.Frame(self.frame_fov)
        self.frame_view = ttk.Frame(self.frame_fov)
        self.frame_pet = ttk.Frame(self.frame_fov)
        self.frame_ct = ttk.Frame(self.frame_fov)
        self.frame_binary = ttk.Frame(self.frame_fov)
        #
        self.frame_slice.pack(fill=X, padx=3, pady=3)
        self.frame_view.pack(fill=X, padx=3, pady=3)
        self.frame_pet.pack(fill=X, padx=3, pady=3)
        self.frame_ct.pack(fill=X, padx=3, pady=3)
        self.frame_binary.pack(fill=X, padx=3, pady=3)

        #frame_slice
        self.entry_slice = ttk.Entry(self.frame_slice, textvariable=self.slice_number, width=5)
        self.button_back = ttk.Button(self.frame_slice, text='<', command=self.backSlice, width=3)
        self.button_next = ttk.Button(self.frame_slice, text='>', command=self.nextSlice, width=3)
        self.button_zoomin = ttk.Button(self.frame_slice, text='+', command=self.zoomIn, width=3)
        self.button_zoomout = ttk.Button(self.frame_slice, text='-', command=self.zoomOut, width=3)
        self.button_reset = ttk.Button(self.frame_slice, text='Reset', command=self.reset)
        #
        self.entry_slice.bind('<Return>', self.sliceByNumber)
        #
        self.entry_slice.pack(side=LEFT)
        self.button_back.pack(side=LEFT)
        self.button_next.pack(side=LEFT)
        self.button_zoomin.pack(side=LEFT)
        self.button_zoomout.pack(side=LEFT)
        self.button_reset.pack(side=LEFT)

        #frame_view
        self.radiobutton_axial = ttk.Radiobutton(self.frame_view, text='Axial', value='axial',
        variable=self.image_view, command=self.changeView)
        self.radiobutton_coronal = ttk.Radiobutton(self.frame_view, text='Coronal', value='coronal',
        variable=self.image_view, command=self.changeView)
        self.radiobutton_sagital = ttk.Radiobutton(self.frame_view, text='Sagital', value='sagital',
        variable=self.image_view, command=self.changeView)
        #
        self.radiobutton_axial.pack(side=LEFT)
        self.radiobutton_coronal.pack(side=LEFT)
        self.radiobutton_sagital.pack(side=LEFT)

        #frame_pet
        self.chechbutton_pet = ttk.Checkbutton(self.frame_pet, text="Pet", width=6, variable=self.pet_show,
        command=self.changePetShow)
        self.scale_pet = ttk.Scale(self.frame_pet, orient=HORIZONTAL, from_=0, to=255, variable=self.pet_opacity,
        command=self.changePetOpacity)
        self.radiobutton_pet_original = ttk.Radiobutton(self.frame_pet, text='', value='original',
        variable=self.pet_color, command=self.changePetColor)
        self.radiobutton_pet_inverse = ttk.Radiobutton(self.frame_pet, text='', value='inverse',
        variable=self.pet_color, command=self.changePetColor)
        self.radiobutton_pet_fire = ttk.Radiobutton(self.frame_pet, text='', value='fire',
        variable=self.pet_color, command=self.changePetColor)
        #
        self.chechbutton_pet.pack(side=LEFT)
        self.scale_pet.pack(side=LEFT)
        self.radiobutton_pet_original.pack(side=LEFT)
        self.radiobutton_pet_inverse.pack(side=LEFT)
        self.radiobutton_pet_fire.pack(side=LEFT)

        #frame_ct
        self.chechbutton_ct = ttk.Checkbutton(self.frame_ct, text="Ct", width=6, variable=self.ct_show,
        command=self.changeCtShow)
        self.scale_ct = ttk.Scale(self.frame_ct, orient=HORIZONTAL, from_=0, to=255, variable=self.ct_opacity,
        command=self.changeCtOpacity)
        self.radiobutton_ct_original = ttk.Radiobutton(self.frame_ct, text='', value='original',
        variable=self.ct_color, command=self.changeCtColor)
        self.radiobutton_ct_inverse = ttk.Radiobutton(self.frame_ct, text='', value='inverse',
        variable=self.ct_color, command=self.changeCtColor)
        #
        self.chechbutton_ct.pack(side=LEFT)
        self.scale_ct.pack(side=LEFT)
        self.radiobutton_ct_original.pack(side=LEFT)
        self.radiobutton_ct_inverse.pack(side=LEFT)

        #frame_binary
        self.chechbutton_binary = ttk.Checkbutton(self.frame_binary, text="Binary", width=6,
        variable=self.binary_show, command=self.changeBinaryShow)
        self.scale_binary = ttk.Scale(self.frame_binary, orient=HORIZONTAL, from_=0, to=255,
        variable=self.binary_opacity, command=self.changeBinaryOpacity)
        self.radiobutton_binary_original = ttk.Radiobutton(self.frame_binary, text='', value='original',
        variable=self.binary_color, command=self.changeBinaryColor)
        self.radiobutton_binary_inverse = ttk.Radiobutton(self.frame_binary, text='', value='inverse',
        variable=self.binary_color, command=self.changeBinaryColor)
        self.radiobutton_binary_mask = ttk.Radiobutton(self.frame_binary, text='', value='mask',
        variable=self.binary_color, command=self.changeBinaryColor)
        self.radiobutton_binary_anomality = ttk.Radiobutton(self.frame_binary, text='', value='anomality',
        variable=self.binary_color, command=self.changeBinaryColor)
        #
        self.chechbutton_binary.pack(side=LEFT)
        self.scale_binary.pack(side=LEFT)
        self.radiobutton_binary_original.pack(side=LEFT)
        self.radiobutton_binary_inverse.pack(side=LEFT)
        self.radiobutton_binary_mask.pack(side=LEFT)
        self.radiobutton_binary_anomality.pack(side=LEFT)

        #frame_ct_window
        self.label_ct_window = ttk.Label(self.frame_ct_window, text='   Ct Window   ')
        self.chechbutton_ct_window = ttk.Checkbutton(self.frame_ct_window, text="",
        variable=self.ct_window, command=self.changeCtWindow)
        self.label_ct_window_min = ttk.Label(self.frame_ct_window, text='   Min')
        self.entry_ct_window_min = ttk.Entry(self.frame_ct_window, textvariable=self.ct_window_min, width=5)
        self.label_ct_window_max = ttk.Label(self.frame_ct_window, text='   Max')
        self.entry_ct_window_max = ttk.Entry(self.frame_ct_window, textvariable=self.ct_window_max, width=5)
        #
        self.entry_ct_window_min.bind('<Return>', self.changeCtWindowMin)
        self.entry_ct_window_max.bind('<Return>', self.changeCtWindowMax)
        #
        self.label_ct_window.pack(side=LEFT)
        self.chechbutton_ct_window.pack(side=LEFT)
        self.label_ct_window_min.pack(side=LEFT)
        self.entry_ct_window_min.pack(side=LEFT)
        self.label_ct_window_max.pack(side=LEFT)
        self.entry_ct_window_max.pack(side=LEFT)

        #frame_threshold
        self.label_threshold = ttk.Label(self.frame_threshold, text='   Threshold   ')
        self.scale_threshold = ttk.Scale(self.frame_threshold, orient=HORIZONTAL, from_=0, to=100, length=100,
        variable=self.binary_threshold, command=self.changeBinaryThreshold)
        self.entry_threshold = ttk.Entry(self.frame_threshold, textvariable=self.binary_threshold, width=5)
        self.label_labaling_percent = ttk.Label(self.frame_threshold, text='%   ')
        self.button_increase = ttk.Button(self.frame_threshold, text='+', width=2,
        command=self.increaseBinaryThreshold)
        self.button_decrease = ttk.Button(self.frame_threshold, text='-', width=2,
        command=self.decreaseBinaryThreshold)
        #
        self.entry_threshold.bind('<Return>', self.changeBinaryThresholdWithEntry)
        #
        self.label_threshold.pack(side=LEFT)
        self.scale_threshold.pack(side=LEFT)
        self.entry_threshold.pack(side=LEFT)
        self.label_labaling_percent.pack(side=LEFT)
        self.button_increase.pack(side=LEFT)
        self.button_decrease.pack(side=LEFT)

        #frame_label_picker
        self.label_label_picker = ttk.Label(self.frame_label_picker, text='   Label Picker   ')
        self.radiobutton_label_tumoral = ttk.Radiobutton(self.frame_label_picker, text='Tumoral',
        value='tumoral', variable=self.label_in_use)
        self.radiobutton_label_suspect = ttk.Radiobutton(self.frame_label_picker, text='Suspect',
        value='suspect', variable=self.label_in_use)
        #
        self.label_label_picker.pack(side=LEFT)
        self.radiobutton_label_tumoral.pack(side=LEFT)
        self.radiobutton_label_suspect.pack(side=LEFT)

        #frame_2d_labeler
        self.radiobutton_2d_labeler = ttk.Radiobutton(self.frame_2d_labeler, text='2D Labeler   |   ',
        value='2d-labeler', variable=self.tool_in_hand)
        self.radiobutton_2d_max_growth = ttk.Radiobutton(self.frame_2d_labeler, text='MaxG',
        value='max', variable=self._2d_labeler_growth)
        self.radiobutton_2d_min_growth = ttk.Radiobutton(self.frame_2d_labeler, text='MinG',
        value='min', variable=self._2d_labeler_growth)
        #
        self.radiobutton_2d_labeler.pack(side=LEFT)
        self.radiobutton_2d_max_growth.pack(side=LEFT)
        self.radiobutton_2d_min_growth.pack(side=LEFT)

        #frame_3d_labeler
        self.radiobutton_3d_labeler = ttk.Radiobutton(self.frame_3d_labeler, text='3D Labeler   |   ',
        value='3d-labeler', variable=self.tool_in_hand)
        self.radiobutton_3d_max_growth = ttk.Radiobutton(self.frame_3d_labeler, text='MaxG',
        value='max', variable=self._3d_labeler_growth)
        self.radiobutton_3d_med_growth = ttk.Radiobutton(self.frame_3d_labeler, text='MedG',
        value='med', variable=self._3d_labeler_growth)
        self.radiobutton_3d_min_growth = ttk.Radiobutton(self.frame_3d_labeler, text='MinG',
        value='min', variable=self._3d_labeler_growth)
        #
        self.radiobutton_3d_labeler.pack(side=LEFT)
        self.radiobutton_3d_max_growth.pack(side=LEFT)
        self.radiobutton_3d_med_growth.pack(side=LEFT)
        self.radiobutton_3d_min_growth.pack(side=LEFT)

        #frame_roi
        self.radiobutton_roi = ttk.Radiobutton(self.frame_roi, text='Roi   |   ', value='roi',
        variable=self.tool_in_hand)
        self.entry_radius = ttk.Entry(self.frame_roi, textvariable=self.roi_radius, width=5)
        self.label_roi_mm = ttk.Label(self.frame_roi, text='mm   ')
        self.label_ump = ttk.Label(self.frame_roi, textvariable=self.roi_ump)
        self.label_roi_percent = ttk.Label(self.frame_roi, text='%   ')
        self.button_to_threshold = ttk.Button(self.frame_roi, text='To Threshold',
        command=self.umpToTreshold)
        #
        self.entry_radius.bind('<Return>', self.changeRoiRadius)
        #
        self.radiobutton_roi.pack(side=LEFT)
        self.entry_radius.pack(side=LEFT)
        self.label_roi_mm.pack(side=LEFT)
        self.label_ump.pack(side=LEFT)
        self.label_roi_percent.pack(side=LEFT)
        self.button_to_threshold.pack(side=LEFT)

        #frame_undo_redo
        self.button_undo = ttk.Button(self.frame_undo_redo, text='Undo', command=self.undo, width=5)
        self.button_redo = ttk.Button(self.frame_undo_redo, text='Redo', command=self.redo, width=5)
        #
        self.button_undo.pack(side=LEFT, padx=3, pady=3)
        self.button_redo.pack(side=LEFT, padx=3, pady=3)

        #frame_info
        self.text_info = Text(self.frame_info, width=1, bg='#444444', fg='white', font=('Courier', '12'),
        state='disable')
        #
        self.text_info.pack(fill=BOTH, expand=True)

        #frame_desk
        self.scrolledframe = ScrolledFrame(self.frame_desk)
        self.scrolledframe.pack(side="top", expand=1, fill="both")
        self.scrolledframe.bind_arrow_keys(self.frame_desk)
        self.scrolledframe.bind_scroll_wheel(self.frame_desk)
        self.scrolledframe_inner_frame = self.scrolledframe.display_widget(Frame)
        #
        self.label_image = ttk.Label(self.scrolledframe_inner_frame)
        #
        self.label_image.bind('<Double-Button-1>', self.clickOnImage)
        self.label_image.bind('<MouseWheel>', self.mousewheelOnImage)
        #
        self.label_image.grid(padx=10, pady=10)

        self.setState(False)
        self.setInfo('welcome :)')

        return

    def updateFrame(self):
        zoom_step = 1.25
        #
        img_shape = self.entities['pet']['slice'].shape
        img_x = self.entities['pet']['x']
        img_y = self.entities['pet']['y']
        img_shape = (round(img_shape[1]*img_y), round(img_shape[0]*img_x))
        #
        if self.zoom_level == 0:
            pass
        elif self.zoom_level < 0:
            temp = zoom_step ** -self.zoom_level
            img_shape = (int(img_shape[0]/temp), int(img_shape[1]/temp))
        else:
            temp = zoom_step ** self.zoom_level
            img_shape = (int(img_shape[0]*temp), int(img_shape[1]*temp))
        #
        img = Image.new('RGBA', img_shape, (0, 0, 0))

        if self.pet_show.get():
            pet_slice = self.entities['pet']['slice']
            pet_x = self.entities['pet']['x']
            pet_y = self.entities['pet']['y']
            pet_shape = pet_slice.shape
            pet_shape = (round(pet_shape[1]*pet_y), round(pet_shape[0]*pet_x))
            #
            if self.zoom_level == 0:
                pass
            elif self.zoom_level < 0:
                temp = zoom_step ** -self.zoom_level
                pet_shape = (int(pet_shape[0]/temp), int(pet_shape[1]/temp))
            else:
                temp = zoom_step ** self.zoom_level
                pet_shape = (int(pet_shape[0]*temp), int(pet_shape[1]*temp))
            #
            img_pet = None
            if self.pet_color.get() == 'original':
                img_pet = Image.fromarray((pet_slice*255).astype(np.int32)).convert('RGB')
            elif self.pet_color.get() == 'inverse':
                img_pet = Image.fromarray(((pet_slice*-255)+255).astype(np.int32)).convert('RGB')
            elif self.pet_color.get() == 'fire':
                img_pet = Image.fromarray((pet_slice*255).astype(np.int32)).convert('P')
                img_pet.putpalette(self.fire_palette)
                img_pet = img_pet.convert('RGB')
            #
            img_pet = img_pet.resize(pet_shape)
            #
            img_pet.putalpha(self.pet_opacity.get())
            img = Image.alpha_composite(img, img_pet)

        if self.ct_show.get():
            ct_slice = self.entities['ct']['slice']
            ct_x = self.entities['ct']['x']
            ct_y = self.entities['ct']['y']
            ct_shape = ct_slice.shape
            ct_shape = (round(ct_shape[1]*ct_y), round(ct_shape[0]*ct_x))
            #
            if self.zoom_level == 0:
                pass
            elif self.zoom_level < 0:
                temp = zoom_step ** -self.zoom_level
                ct_shape = (int(ct_shape[0]/temp), int(ct_shape[1]/temp))
            else:
                temp = zoom_step ** self.zoom_level
                ct_shape = (int(ct_shape[0]*temp), int(ct_shape[1]*temp))
            #
            img_ct = None
            if self.ct_color.get() == 'original':
                img_ct = Image.fromarray((ct_slice*255).astype(np.int32)).convert('RGB')
            elif self.ct_color.get() == 'inverse':
                img_ct = Image.fromarray(((ct_slice*-255)+255).astype(np.int32)).convert('RGB')
            #
            img_ct = img_ct.resize(ct_shape)
            m_v = img_shape[1] - ct_shape[1]
            m_t = int(m_v/2)
            m_b = int(m_v-m_t)
            m_h = img_shape[0] - ct_shape[0]
            m_r = int(m_h/2)
            m_l = int(m_h-m_r)
            if self.ct_color.get() == 'original':
                img_ct = self.addMargin(img_ct, m_t, m_r, m_b, m_l, (0,0,0))
            elif self.ct_color.get() == 'inverse':
                img_ct = self.addMargin(img_ct, m_t, m_r, m_b, m_l, (255,255,255))
            #
            img_ct.putalpha(self.ct_opacity.get())
            img = Image.alpha_composite(img, img_ct)

        if self.binary_show.get():
            binary_slice = self.entities['binary']['slice']
            binary_shape = binary_slice.shape
            #
            img_binary = None
            if self.binary_color.get() == 'original':
                img_binary = Image.fromarray((binary_slice*255).astype(np.int32)).convert('RGB')
                img_binary.putalpha(self.binary_opacity.get())
            elif self.binary_color.get() == 'inverse':
                img_binary = Image.fromarray(((binary_slice*-255)+255).astype(np.int32)).convert('RGB')
                img_binary.putalpha(self.binary_opacity.get())
            elif self.binary_color.get() == 'mask':
                img_binary = Image.fromarray((binary_slice*255).astype(np.int32)).convert('P')
                img_binary.putpalette(self.mask_palette)
                img_binary = img_binary.convert('RGB')
                binary_opacity = self.binary_opacity.get()
                img_binary.putalpha(Image.fromarray((binary_slice*binary_opacity).astype(np.int32)).convert('L'))
            elif self.binary_color.get() == 'anomality':
                anomality_slice = self.entities['anomality']['slice']
                temp = ((binary_slice*0.25) + anomality_slice) / 1.25
                img_binary = Image.fromarray((temp*255).astype(np.int32)).convert('P')
                img_binary.putpalette(self.anomality_palette)
                img_binary = img_binary.convert('RGB')
                binary_opacity = self.binary_opacity.get()
                temp = np.where(temp>0, binary_opacity, 0)
                img_binary.putalpha(Image.fromarray(temp).convert('L'))
            #
            binary_shape = img_shape
            img_binary = img_binary.resize(binary_shape)
            img = Image.alpha_composite(img, img_binary)

        if self.roi_show:
            roi_width_in_pixel = int((self.roi_radius.get()/1)*2)
            if self.zoom_level == 0:
                pass
            elif self.zoom_level < 0:
                temp = zoom_step ** -self.zoom_level
                roi_width_in_pixel = int(roi_width_in_pixel/temp)
            else:
                temp = zoom_step ** self.zoom_level
                roi_width_in_pixel = int(roi_width_in_pixel*temp)
            img_roi = self.roi_image.resize((roi_width_in_pixel, roi_width_in_pixel))
            temp = (self.roi_coordinate[0]-round(roi_width_in_pixel/2),
            self.roi_coordinate[1]-round(roi_width_in_pixel/2))
            img.paste(img_roi, temp, mask=img_roi.getchannel('A'))

        img = ImageTk.PhotoImage(img)
        self.label_image.configure(image=img)
        self.label_image.image = img
        self.image_shape = img_shape

        return

    def sliceByNumber(self, e):
        self.entities = self.dicom.getSlices(slice_number=self.slice_number.get())
        self.updateFrame()
        self.slice_number.set(self.dicom.current_slice_number)
        return

    def nextSlice(self):
        self.entities = self.dicom.getSlices(slice_number=self.dicom.current_slice_number+1)
        self.updateFrame()
        self.slice_number.set(self.dicom.current_slice_number)
        return

    def backSlice(self):
        self.entities = self.dicom.getSlices(slice_number=self.dicom.current_slice_number-1)
        self.updateFrame()
        self.slice_number.set(self.dicom.current_slice_number)
        return

    def zoomIn(self):
        if self.zoom_level < 3:
            self.zoom_level += 1
            self.updateFrame()
        return

    def zoomOut(self):
        if self.zoom_level > -10:
            self.zoom_level -= 1
            self.updateFrame()
        return

    def reset(self):
        if self.zoom_level != 0:
            self.zoom_level = 0
            self.updateFrame()
        return

    def changeView(self):
        self.entities = self.dicom.getSlices(view=self.image_view.get(), slice_number='middle')
        self.updateFrame()
        self.image_view.set(self.dicom.current_view)
        self.slice_number.set(self.dicom.current_slice_number)
        return

    def changePetShow(self):
        self.updateFrame()
        return

    def changePetOpacity(self, opacity):
        opacity = int(float(opacity))
        self.pet_opacity.set(opacity)
        self.updateFrame()
        return

    def changePetColor(self):
        self.updateFrame()
        return

    def changeCtShow(self):
        self.updateFrame()
        return

    def changeCtOpacity(self, opacity):
        opacity = int(float(opacity))
        self.ct_opacity.set(opacity)
        self.updateFrame()
        return

    def changeCtColor(self):
        self.updateFrame()
        return

    def changeBinaryShow(self):
        self.updateFrame()
        return

    def changeBinaryOpacity(self, opacity):
        opacity = int(float(opacity))
        self.binary_opacity.set(opacity)
        self.updateFrame()
        return

    def changeBinaryColor(self):
        self.updateFrame()
        return
    
    def changeCtWindow(self):
        ct_window = (self.ct_window.get(), self.ct_window_min.get(), self.ct_window_max.get())
        self.entities = self.dicom.getSlices(ct_window=ct_window)
        self.updateFrame()
        return

    def changeCtWindowMin(self, min):
        ct_window = (self.ct_window.get(), self.ct_window_min.get(), self.ct_window_max.get())
        self.entities = self.dicom.getSlices(ct_window=ct_window)
        self.updateFrame()
        return

    def changeCtWindowMax(self, max):
        ct_window = (self.ct_window.get(), self.ct_window_min.get(), self.ct_window_max.get())
        self.entities = self.dicom.getSlices(ct_window=ct_window)
        self.updateFrame()
        return

    def changeBinaryThreshold(self, threshold):
        threshold = int(float(threshold))
        self.binary_threshold.set(threshold)
        self.entities = self.dicom.getSlices(threshold=threshold/100)
        self.updateFrame()
        return

    def changeBinaryThresholdWithEntry(self, event):
        binary_threshold = self.binary_threshold.get()
        binary_threshold = round(binary_threshold,1)
        if binary_threshold < 0:
            binary_threshold = 0
        elif binary_threshold > 100:
            binary_threshold = 100
        self.binary_threshold.set(binary_threshold)
        self.entities = self.dicom.getSlices(threshold=binary_threshold/100)
        self.updateFrame()
        return

    def increaseBinaryThreshold(self):
        binary_threshold = self.binary_threshold.get()
        if binary_threshold < 100:
            binary_threshold = int(binary_threshold + 1)
            self.binary_threshold.set(binary_threshold)
            self.entities = self.dicom.getSlices(threshold=binary_threshold/100)
            self.updateFrame()
        return

    def decreaseBinaryThreshold(self):
        binary_threshold = self.binary_threshold.get()
        if binary_threshold > 0:
            binary_threshold = int(binary_threshold - 1)
            self.binary_threshold.set(binary_threshold)
            self.entities = self.dicom.getSlices(threshold=binary_threshold/100)
            self.updateFrame()
        return

    def changeRoiRadius(self, event):
        roi_radius = self.roi_radius.get()
        if roi_radius < 0:
            roi_radius = 0
            self.roi_radius.set(roi_radius)
        elif roi_radius > 100:
            roi_radius = 100
            self.roi_radius.set(roi_radius)
        return

    def undo(self):
        message = self.dicom.undoRedo('undo')
        self.entities = self.dicom.getSlices()
        self.updateFrame()
        self.setInfo(message)
        return

    def redo(self):
        message = self.dicom.undoRedo('redo')
        self.entities = self.dicom.getSlices()
        self.updateFrame()
        self.setInfo(message)
        return

    def umpToTreshold(self):
        binary_threshold = self.roi_ump.get()
        self.binary_threshold.set(binary_threshold)
        self.entities = self.dicom.getSlices(threshold=binary_threshold/100)
        self.updateFrame()
        return

    def clickOnImage(self, e):
        image_width = self.image_shape[0]
        image_height = self.image_shape[1]
        pet_width = self.entities['pet']['slice'].shape[1]
        pet_height = self.entities['pet']['slice'].shape[0]
        x = int((e.x/image_width)*pet_width)
        y = int((e.y/image_height)*pet_height)
        coordinate = (self.dicom.current_slice_number, y, x)

        if self.tool_in_hand.get() == '2d-labeler':
            message = self.dicom.label2D((coordinate[1], coordinate[2]),
            self.label_in_use.get(), self._2d_labeler_growth.get())
            self.setInfo(message)
            self.entities = self.dicom.getSlices()
            self.updateFrame()
        elif self.tool_in_hand.get() == '3d-labeler':
            message = self.dicom.label(coordinate,
            self.label_in_use.get(), self._3d_labeler_growth.get())
            self.setInfo(message)
            self.entities = self.dicom.getSlices()
            self.updateFrame()
        elif self.tool_in_hand.get() == 'roi':
            self.roi_show = True
            self.roi_coordinate = (e.x, e.y)
            roi_radius = self.roi_radius.get()
            if roi_radius < 0:
                roi_radius = 0
                self.roi_radius.set(roi_radius)
            if roi_radius > 100:
                roi_radius = 100
                self.roi_radius.set(roi_radius)
            (roi_ump, message) = self.dicom.roi(coordinate, roi_radius)
            self.roi_ump.set(roi_ump)
            self.setInfo(message)
            if roi_radius == 0:
                self.roi_show = False
                self.updateFrame()
            else:
                self.updateFrame()
                self.roi_show = False

        return

    def mousewheelOnImage(self, e):
        if e.delta > 0:
            self.nextSlice()
        else:
            self.backSlice()
        return

    def setInfo(self, message):
        self.text_info.config(state='normal')
        self.text_info.delete('1.0', 'end')
        self.text_info.insert('1.0', message)
        self.text_info.config(state='disable')

    def setState(self, state):
        self.state = state
        if self.state:
            self.entry_slice.configure(state='normal')
            self.entry_ct_window_min.configure(state='normal')
            self.entry_ct_window_max.configure(state='normal')
            self.entry_threshold.configure(state='normal')
            self.entry_radius.configure(state='normal')
            #
            self.button_back.configure(state='normal')
            self.button_next.configure(state='normal')
            self.button_zoomin.configure(state='normal')
            self.button_zoomout.configure(state='normal')
            self.button_reset.configure(state='normal')
            self.button_decrease.configure(state='normal')
            self.button_increase.configure(state='normal')
            self.button_to_threshold.configure(state='normal')
            self.button_undo.configure(state='normal')
            self.button_redo.configure(state='normal')
            #
            self.chechbutton_pet.configure(state='normal')
            self.chechbutton_ct.configure(state='normal')
            self.chechbutton_binary.configure(state='normal')
            self.chechbutton_ct_window.configure(state='normal')##
            #
            self.radiobutton_axial.configure(state='normal')
            self.radiobutton_coronal.configure(state='normal')
            self.radiobutton_sagital.configure(state='normal')
            self.radiobutton_pet_original.configure(state='normal')
            self.radiobutton_pet_inverse.configure(state='normal')
            self.radiobutton_pet_fire.configure(state='normal')
            self.radiobutton_ct_original.configure(state='normal')
            self.radiobutton_ct_inverse.configure(state='normal')
            self.radiobutton_binary_original.configure(state='normal')
            self.radiobutton_binary_inverse.configure(state='normal')
            self.radiobutton_binary_mask.configure(state='normal')
            self.radiobutton_binary_anomality.configure(state='normal')
            self.radiobutton_label_tumoral.configure(state='normal')
            self.radiobutton_label_suspect.configure(state='normal')
            self.radiobutton_2d_labeler.configure(state='normal')
            self.radiobutton_2d_max_growth.configure(state='normal')
            self.radiobutton_2d_min_growth.configure(state='normal')
            self.radiobutton_3d_labeler.configure(state='normal')
            self.radiobutton_3d_max_growth.configure(state='normal')
            self.radiobutton_3d_med_growth.configure(state='normal')
            self.radiobutton_3d_min_growth.configure(state='normal')
            self.radiobutton_roi.configure(state='normal')
            #
            self.scale_pet.configure(state='normal')
            self.scale_ct.configure(state='normal')
            self.scale_binary.configure(state='normal')
            self.scale_threshold.configure(state='normal')
        else:
            self.entry_slice.configure(state='disable')
            self.entry_ct_window_min.configure(state='disable')
            self.entry_ct_window_max.configure(state='disable')
            self.entry_threshold.configure(state='disable')
            self.entry_radius.configure(state='disable')
            #
            self.button_back.configure(state='disable')
            self.button_next.configure(state='disable')
            self.button_zoomin.configure(state='disable')
            self.button_zoomout.configure(state='disable')
            self.button_reset.configure(state='disable')
            self.button_decrease.configure(state='disable')
            self.button_increase.configure(state='disable')
            self.button_to_threshold.configure(state='disable')
            self.button_undo.configure(state='disable')
            self.button_redo.configure(state='disable')
            #
            self.chechbutton_pet.configure(state='disable')
            self.chechbutton_ct.configure(state='disable')
            self.chechbutton_binary.configure(state='disable')
            self.chechbutton_ct_window.configure(state='disable')
            #
            self.radiobutton_axial.configure(state='disable')
            self.radiobutton_coronal.configure(state='disable')
            self.radiobutton_sagital.configure(state='disable')
            self.radiobutton_pet_original.configure(state='disable')
            self.radiobutton_pet_inverse.configure(state='disable')
            self.radiobutton_pet_fire.configure(state='disable')
            self.radiobutton_ct_original.configure(state='disable')
            self.radiobutton_ct_inverse.configure(state='disable')
            self.radiobutton_binary_original.configure(state='disable')
            self.radiobutton_binary_inverse.configure(state='disable')
            self.radiobutton_binary_mask.configure(state='disable')
            self.radiobutton_binary_anomality.configure(state='disable')
            self.radiobutton_label_tumoral.configure(state='disable')
            self.radiobutton_label_suspect.configure(state='disable')
            self.radiobutton_2d_labeler.configure(state='disable')
            self.radiobutton_2d_max_growth.configure(state='disable')
            self.radiobutton_2d_min_growth.configure(state='disable')
            self.radiobutton_3d_labeler.configure(state='disable')
            self.radiobutton_3d_max_growth.configure(state='disable')
            self.radiobutton_3d_med_growth.configure(state='disable')
            self.radiobutton_3d_min_growth.configure(state='disable')
            self.radiobutton_roi.configure(state='disable')
            #
            self.scale_pet.configure(state='disable')
            self.scale_ct.configure(state='disable')
            self.scale_binary.configure(state='disable')
            self.scale_threshold.configure(state='disable')
        return

    def open(self):
        if self.state:
            self.save()
        try:
            self.paths = {
                'pet': None,
                'ct': None,
                'label': None
            }
            for key in self.paths.keys():
                messagebox.showinfo(title=key, message='Please select '+key+' folder.')
                folder = filedialog.askdirectory()
                if folder:
                    self.paths[key] = folder
                else:
                    raise Exception('path has not set.')
            self.dicom.open(self.paths['pet'], self.paths['ct'], self.paths['label'])
            self.entities = self.dicom.getSlices('axial', 0, 0.2)
            self.image_view.set(self.dicom.current_view)
            self.slice_number.set(self.dicom.current_slice_number)
            #
            self.pet_show.set(True)
            self.pet_opacity.set(255)
            self.pet_color.set('original')
            self.ct_show.set(False)
            self.ct_opacity.set(127)
            self.ct_color.set('original')
            self.binary_show.set(False)
            self.binary_opacity.set(127)
            self.binary_color.set('mask')
            self.ct_window.set(False)
            self.ct_window_min.set(-130)
            self.ct_window_max.set(270)
            self.binary_threshold.set(20)
            self.label_in_use.set('tumoral')
            self._2d_labeler_growth.set('min')
            self._3d_labeler_growth.set('min')
            self.roi_radius.set(10)
            #
            self.setInfo('new dicom.')
            self.setState(True)
            self.updateFrame()
            messagebox.showinfo(title="Open", message='Opening was successfull.')
        except Exception as e:
            print(e)
            messagebox.showerror(title="Can't Open", message='File is not suitable!')
        return

    def save(self):
        if self.state:
            save = messagebox.askyesno(title='Save', message='Do you want to save file operation?')
            if save:
                try:
                    self.dicom.save(self.paths['label'])
                    messagebox.showinfo(title="Save", message='Saving was successfull.')
                except Exception as e:
                    print(e)
                    messagebox.showerror(title="Can't Save", message='Saving is not applicable!')
        else:
            messagebox.showinfo(title="Save", message='There is nothing for saving!')
        return

    def exit(self):
        if self.state:
            self.save()
        exit = messagebox.askyesno(title='Exit', message='Do you want to exit?')
        if exit:
            self.master.quit()
        return

    #private
    def addMargin(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result


def main():
    root = Tk()
    PltApp(root)
    root.mainloop()


if __name__ == "__main__": main()
