from abaqus import mdb
from abaqus import *
from abaqusConstants import *
from caeModules import *
from part import *
import job
import sys, os
sys.path.append('filepath')
from database import *
from database_output import *
from assemble import assemble
import json


class FEAutomation:
    def __init__(self, folder_path, file_suffix, all_name, pressure_set_file, cf3_set_file):
        self.folder_path = folder_path
        self.file_suffix = file_suffix
        self.all_name = all_name
        self.pressure_set_file = pressure_set_file
        self.cf3_set_file = cf3_set_file

        self.matching_files_list = self.get_matching_files()
        self.pressure_set = self.load_json_file(pressure_set_file)
        self.cf3_values_sets = self.load_json_file(cf3_set_file)

    def get_matching_files(self):
        files = os.listdir(self.folder_path)
        matching_files = [os.path.splitext(file)[0] for file in files if file.endswith(self.file_suffix)]
        return list(matching_files)

    def load_json_file(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def assemble_and_run_jobs(self, assemble):
        for model_idx, model_name in enumerate(self.all_name):
            pressure = self.pressure_set[model_idx]
            cf3_values = self.cf3_values_sets[model_idx // 3]

            assemble(model_set=model_name, pressure=pressure, cf3_values=cf3_values)

            job_name = "Job_{}".format(model_name)
            myJob = mdb.Job(name=job_name, model=model_name, description='', type=ANALYSIS,
                            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                            scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1,
                            multiprocessingMode=DEFAULT, numCpus=4, numDomains=4, numGPUs=0)

            myJob.submit()
            myJob.waitForCompletion()
            
            # monitoring results
            print(job_name + 'Succeed')



# Usage
folder_path = r'filepath'
file_suffix = '.cae'
all_name =[
           '01_processed']
pressure_set_file = 'pressure_set.json'
cf3_set_file = 'cf3_set_new.json'

fe_automation = FEAutomation(folder_path, file_suffix, all_name, pressure_set_file, cf3_set_file)
fe_automation.assemble_and_run_jobs(assemble)
