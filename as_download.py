'''
================================================
          DOWNLOAD_AUDIOSET REPOSITORY
================================================
Original:
    repository name: download_audioset
    repository version: 1.0
    repository link: https://github.com/jim-schwoebel/download_audioset
    author: Jim Schwoebel
    author contact: js@neurolex.co
    description: downloads the raw audio files from AudioSet (released by Google).
    license category: opensource
    license: Apache 2.0 license
    organization name: NeuroLex Laboratories, Inc.
    location: Seattle, WA
    website: https://neurolex.ai
    release date: 2018-11-08

Edit:
    repository name: download_audioset
    repository version: 1.1
    repository link: https://github.com/frozenburst/download_audioset
    author: POYU WU
    release date: 2020-11-10

This code (download_audioset) is hereby released under a Apache 2.0 license license.

For more information, check out the license terms below.

================================================
                SPECIAL NOTES
================================================

This script parses through the entire balanced audioset dataset and downloads
all the raw audio files. The files are arranged in folders according to their
representative classes.

Please ensure that you have roughly 35GB of free space on your computer before
downloading the files. Note that it may take up to 2 days to fully download
all the files.

Enjoy! - :)

#-Jim

================================================
                LICENSE TERMS
================================================

Copyright 2018 NeuroLex Laboratories, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

================================================
                SERVICE STATEMENT
================================================

If you are using the code written for a larger project, we are
happy to consult with you and help you with deployment. Our team
has >10 world experts in Kafka distributed architectures, microservices
built on top of Node.js / Python / Docker, and applying machine learning to
model speech and text data.

We have helped a wide variety of enterprises - small businesses,
researchers, enterprises, and/or independent developers.

If you would like to work with us let us know @ develop@neurolex.co.

usage: as_download.py  [options]

options:
    --data_pth=<data path>
    --label_pth=<labels.xlsx>
    --segment_file=<xlsx file>
    --partial=<0, 1, 2, ...> # The unbalance csv could split to parts for parallel.
'''

################################################################################
##                            IMPORT STATEMENTS                               ##
################################################################################

import pafy, os, shutil, time, ffmpy
import os.path as op
import pandas as pd
import soundfile as sf

from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
from docopt import docopt

################################################################################
##                            HELPER FUNCTIONS                                ##
################################################################################

#function to clean labels
def convertlabels(sortlist,labels,textlabels):

    clabels=list()
    # Debug for sortlist data type, split with each label ids.
    sortlist = sortlist.split(',')

    for i in range(len(sortlist)):
        #find index in list corresponding
        index=labels.index(sortlist[i])
        clabel=textlabels[index]
        #pull out converted label
        clabels.append(clabel)

    return clabels


def download_audio(link):
    listdir=os.listdir()
    cmd = f"youtube-dl --quiet -f 'bestaudio[ext=m4a]' '{link}'"
    print(cmd)
    os.system(cmd)
    listdir2=os.listdir()
    filename=''
    for i in range(len(listdir2)):
        if listdir2[i] not in listdir and listdir2[i].endswith('.m4a'):
            filename=listdir2[i]
            break

    return filename

################################################################################
##                            MAIN SCRIPT                                     ##
################################################################################

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    data_pth = args['--data_pth']
    label_pth = args['--label_pth']
    segment_file = args['--segment_file']
    partial = args['--partial']

    if data_pth is None:
        raise ValueError("Please set the path for model's output.")
    if label_pth is None:
        raise ValueError("Please set the path for model's output.")
    if segment_file is None:
        raise ValueError("Please set the path for model's output.")
    if partial is not None:
        print("Partial detected. The naming of wav would follow the partial name.")
    
    defaultdir=os.getcwd()
    os.chdir(defaultdir)

    #load labels of the videos

    #number, label, words
    loadfile=pd.read_excel(label_pth)

    number=loadfile.iloc[:,0].tolist()
    labels=loadfile.iloc[:,1].tolist()
    textlabels=loadfile.iloc[:,2].tolist()
    #remove spaces for folders
    for i in range(len(textlabels)):
        textlabels[i]=textlabels[i].replace(' ','')

    #now load data for download
    xlsx_filename = segment_file
    if op.isfile(xlsx_filename) is False:
        raise ValueError("Xlsx file of segment is not exits with value:", xlsx_filename) 
    loadfile2=pd.read_excel(xlsx_filename)

    # ylabels have to be cleaned to make a good list (CSV --> LIST)
    yid=loadfile2.iloc[:,0].tolist()[2:]
    ystart=loadfile2.iloc[:,1].tolist()[2:]
    yend=loadfile2.iloc[:,2].tolist()[2:]
    ylabels=loadfile2.iloc[:,3].tolist()[2:]

    dataset_dir = data_pth
    if op.isdir(dataset_dir) is False:
        raise ValueError("Dataset directory is not exits with path:", dataset_dir)

    #make folders
    if partial is not None:
        # segment_folder_name = op.basename(xlsx_filename).split('.')[0]
        # Easy method is the best solution.
        segment_folder_name = 'unbalanced_train_segments'
    else:
        segment_folder_name = op.basename(xlsx_filename).split('.')[0]
    try:
        defaultdir2=op.join(dataset_dir, segment_folder_name)
        os.chdir(defaultdir2)
    except:
        defaultdir2=op.join(dataset_dir, segment_folder_name)
        os.mkdir(defaultdir2)
        os.chdir(defaultdir2)

    # Should implement the check of existed file as well.
    # Implemented by frozenburst
    existing_wavfiles=list()
    for dirname in tqdm(sorted(Path(defaultdir2).glob('*'))):
        if partial is not None:
            for filename in sorted(Path(dirname).glob(f'{partial}_*')):
                existing_wavfiles.append(op.basename(filename))
        else:
            for filename in sorted(Path(dirname).glob(f'*')):
                existing_wavfiles.append(op.basename(filename))

    # get last file checkpoint to leave off
    existing_wavfiles=natsorted(existing_wavfiles)
    print(existing_wavfiles)
    try:
        lastfile=int(existing_wavfiles[-1].split('.')[0][7:])
    except:
        lastfile=0

    #iterate through entire CSV file, look for '--' if found, find index, delete section, then go to next index
    slink='https://www.youtube.com/watch?v='

    for i in tqdm(range(len(yid))):
        if i < lastfile:
            # print('Skipping, already downloaded file...')
            continue
        else:
            link=slink+yid[i]
            start=float(ystart[i])
            end=float(yend[i])
            # print(ylabels[i])
            clabels=convertlabels(ylabels[i],labels,textlabels)
            # print(clabels)

            if clabels != []:
                #change to the right directory
                for j in range(len(clabels)):
                    newdir = op.join(defaultdir2, clabels[j])
                    if op.isdir(newdir) is False:
                        os.mkdir(newdir)
                    os.chdir(newdir)
                    #if it is the first download, pursue this path to download video
                    lastdir=os.getcwd()

                    if partial is not None:
                        filename_check = f'{partial}_snipped'+str(i)+'.wav'
                    else:
                        filename_check = 'snipped'+str(i)+'.wav'

                    if filename_check not in os.listdir():
                        try:
                            # use YouTube DL to download audio
                            filename=download_audio(link)
                            extension='.m4a'
                            #get file extension and convert to .wav for processing later
                            os.rename(filename,'%s%s'%(str(i),extension))
                            filename='%s%s'%(str(i),extension)
                            if extension not in ['.wav']:
                                xindex=filename.find(extension)
                                filename=filename[0:xindex]
                                ff=ffmpy.FFmpeg(
                                    inputs={filename+extension:None},
                                    outputs={filename+'.wav':None}
                                    )
                                ff.run()
                                os.remove(filename+extension)

                            file=filename+'.wav'
                            data,samplerate=sf.read(file)
                            totalframes=len(data)
                            totalseconds=totalframes/samplerate
                            startsec=start
                            startframe=samplerate*startsec
                            endsec=end
                            endframe=samplerate*endsec
                            # print(startframe)
                            # print(endframe)
                            if partial is not None:
                                newname = f'{partial}_snipped'+file
                            else:
                                newname = 'snipped'+file
                            sf.write(newname, data[int(startframe):int(endframe)], samplerate)
                            snippedfile=newname
                            os.remove(file)

                        except:
                            print('no urls')

                    #sleep 3 second sleep to prevent IP from getting banned
                        time.sleep(2)
                    else:
                        print('skipping, already downloaded file...')

