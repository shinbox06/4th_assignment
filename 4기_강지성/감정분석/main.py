from Aurora3 import Aurora3 as A3
import pandas as pd

def main():
    df = pd.read_csv("text_data.text", sep="\t", header=None)
    df.columns = ['sentence']
    
    sent_dic = pd.read_csv('SentiWord_Dict.txt',sep = '\t',header=None)
    sent_dic.iloc[0,0]='갈등'

    pos_dic = sent_dic[sent_dic[1]>0]
    neg_dic = sent_dic[sent_dic[1]<0]
    neu_dic = sent_dic[sent_dic[1]==0]
    
    test = A3(df,sent_dic)
    res = test.get_df()
    print(res)

if __name__ == '__main__':
    main()