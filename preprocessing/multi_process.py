#*******************************************************************************
#
#	© Copyright 2021, James Owler
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <https://www.gnu.org/licenses/>.
#  
#*******************************************************************************

import glob
import os
import shutil

from preprocess import preprocessing, contrast_enhancement

def multi_process(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*')):
        print(i)
        preprocessing(i)

def rename(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*')):
        i_new = i.replace('-processed', '')
        print(i_new)
        shutil.move(i, i_new)

def multi_process_contrast(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*.png')):
        print(i)
        preprocessing(i)
        contrast_enhancement(i)

if __name__ == '__main__':
    multi_process_contrast(r"C:\Users\James\Projects\project\data\DRIVE\imgs-clahe-norm")
