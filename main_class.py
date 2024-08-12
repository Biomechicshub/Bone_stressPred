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
           '01_processed', '01_processed02', '01_processed03', '03_processed', '03_processed02', '03_processed03', 
           '04_processed', '04_processed02', '04_processed03', '05_processed', '05_processed02', '05_processed03', 
           '06_processed', '06_processed02', '06_processed03', '07_processed', '07_processed02', '07_processed03', 
           '08_processed', '08_processed02', '08_processed03', '09_processed', '09_processed02', '09_processed03', 
           '10_processed', '10_processed02', '10_processed03', '11_processed', '11_processed02', '11_processed03', 
           '12_processed', '12_processed02', '12_processed03', '13_processed', '13_processed02', '13_processed03', 
           '14_processed', '14_processed02', '14_processed03', '15_processed', '15_processed02', '15_processed03', 
           '16_processed', '16_processed02', '16_processed03', '17_processed', '17_processed02', '17_processed03', 
           '18_processed', '18_processed02', '18_processed03', '19_processed', '19_processed02', '19_processed03', 
           '20_processed', '20_processed02', '20_processed03', '21_processed', '21_processed02', '21_processed03', 
           '22_processed', '22_processed02', '22_processed03', '23_processed', '23_processed02', '23_processed03', 
           '24_processed', '24_processed02', '24_processed03', '25_processed', '25_processed02', '25_processed03', 
           '26_processed', '26_processed02', '26_processed03', '27_processed', '27_processed02', '27_processed03', 
           '28_processed', '28_processed02', '28_processed03', '29_processed', '29_processed02', '29_processed03', 
           '30_processed', '30_processed02', '30_processed03', '31_processed', '31_processed02', '31_processed03', 
           '32_processed', '32_processed02', '32_processed03', '33_processed', '33_processed02', '33_processed03', 
           '34_processed', '34_processed02', '34_processed03', '35_processed', '35_processed02', '35_processed03', 
           '36_processed', '36_processed02', '36_processed03', '37_processed', '37_processed02', '37_processed03', 
           '38_processed', '38_processed02', '38_processed03', '39_processed', '39_processed02', '39_processed03', 
           '40_processed', '40_processed02', '40_processed03', '41_processed', '41_processed02', '41_processed03', 
           '42_processed', '42_processed02', '42_processed03', '43_processed', '43_processed02', '43_processed03', 
           '44_processed', '44_processed02', '44_processed03', '45_processed', '45_processed02', '45_processed03', 
           '46_processed', '46_processed02', '46_processed03', '47_processed', '47_processed02', '47_processed03', 
           '48_processed', '48_processed02', '48_processed03', '49_processed', '49_processed02', '49_processed03', 
           '50_processed', '50_processed02', '50_processed03', '51_processed', '51_processed02', '51_processed03']
pressure_set_file = 'pressure_set.json'
cf3_set_file = 'cf3_set_new.json'

fe_automation = FEAutomation(folder_path, file_suffix, all_name, pressure_set_file, cf3_set_file)
fe_automation.assemble_and_run_jobs(assemble)
