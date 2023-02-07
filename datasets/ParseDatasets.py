"""
@author Liu Lei
"""
from scipy import io

class DataSetBase(object):
    def __init__(self):
        super().__init__()

    def get_data_path(self):
        raise NotImplementedError

    def parse_data_from_path(self):
        raise NotImplementedError
class CWRU(DataSetBase):
    def __init__(self):
        super().__init__()

    def get_data_path(self):
        normal = ['normal_0_097','normal_1_098','normal_2_099','normal_3_100']
        ball7 = ['12k_Drive_End_B007_0_118','12k_Drive_End_B007_1_119','12k_Drive_End_B007_2_120','12k_Drive_End_B007_3_121']
        ball14 = ['12k_Drive_End_B014_0_185',"12k_Drive_End_B014_1_186","12k_Drive_End_B014_2_187","12k_Drive_End_B014_3_188"]
        ball21 = ['12k_Drive_End_B021_0_222','12k_Drive_End_B021_1_223','12k_Drive_End_B021_2_224','12k_Drive_End_B021_3_225']
        inner7 = ['12k_Drive_End_IR007_0_105','12k_Drive_End_IR007_1_106','12k_Drive_End_IR007_2_107','12k_Drive_End_IR007_3_108']
        inner14 = ['12k_Drive_End_IR014_0_169','12k_Drive_End_IR014_1_170','12k_Drive_End_IR014_2_171','12k_Drive_End_IR014_3_172']
        inner21 = ['12k_Drive_End_IR021_0_209','12k_Drive_End_IR021_1_210','12k_Drive_End_IR021_2_211','12k_Drive_End_IR021_3_212']
        outer7 = ['12k_Drive_End_OR007@6_0_130','12k_Drive_End_OR007@6_1_131','12k_Drive_End_OR007@6_2_132','12k_Drive_End_OR007@6_3_133']
        outer14 = ['12k_Drive_End_OR014@6_0_197','12k_Drive_End_OR014@6_1_198','12k_Drive_End_OR014@6_2_199','12k_Drive_End_OR014@6_3_200']
        outer21 = ['12k_Drive_End_OR021@6_0_234','12k_Drive_End_OR021@6_1_235','12k_Drive_End_OR021@6_2_236','12k_Drive_End_OR021@6_3_237']
        all_data_path = [normal,ball7,ball14,ball21,inner7,inner14,inner21,outer7,outer14,outer21]
        return all_data_path

    def parse_data_from_path(self,hp_path):
        base_path = 'D:/Datas/CWRU/'
        num = hp_path.split('_')[-1]
        path = base_path + hp_path
        data = io.loadmat(path)["X" + num + "_DE_time"]
        return data.squeeze()
