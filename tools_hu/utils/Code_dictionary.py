import pandas as pd


class CodeDictionary():
    def __init__(self, code_file, code_id=None):
        self.code_df = pd.read_table(code_file, header=-1, names=['CODE'], encoding='gbk')
        if code_id is not None:
            self.id_df = pd.read_table(code_id, header=-1, names=['ID'], encoding='gbk')
        else:
            self.id_df = None
        if self.id_df is None:
            self.code_df['ID'] = self.code_df.index + 1
        else:
            self.code_df['ID'] = self.id_df['ID']
        print(self.code_df)
        self.code_dict, self.code_list, self.code_id_lst = self.get_dict(self.code_df)

    def code2id(self, code_lst):
        if isinstance(code_lst, str):
            return self.code_df.loc[self.code_df['CODE'] == code_lst, 'ID'].values[0]
        if isinstance(code_lst, list):
            id_lst = []
            for code in code_lst:
                id_lst.append(self.code_df.loc[self.code_df['CODE'] == code, 'ID'].values[0])
            return id_lst

    def id2code(self, id_lst):
        if isinstance(id_lst, int):
            return self.code_df.loc[self.code_df['ID'] == id_lst, 'CODE'].values[0]
        if isinstance(id_lst, list):
            code_lst = []
            for id in id_lst:
                code_lst.append(self.code_df.loc[self.code_df['ID'] == id, 'CODE'].values[0])
            return code_lst

    def idx2code(self, idx):
        if isinstance(idx, int):
            return self.code_df.loc[idx, 'CODE']

    @staticmethod
    def get_dict(code_df):
        d_ = {}
        code_lst = []
        code_id_lst = []
        for i in range(len(code_df)):
            d_.setdefault(code_df.loc[i, 'CODE'], code_df.loc[i, 'ID'])
            code_lst.append(code_df.loc[i, 'CODE'])
            code_id_lst.append(str(code_df.loc[i, 'ID']))
        return d_, code_lst, code_id_lst


if __name__ == '__main__':
    code_f = r'/data/sdv1/whtm/document/G6_21101-V1.0.txt'
    code = CodeDictionary(code_f)
    print(code.code_list)
