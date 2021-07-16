import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import codecs
import time
import csv
import jieba.analyse
import codecs
time_begin=time.time()
txt = open("shuju.txt", 'rt', encoding = 'utf-8').read()  #读取所需要分析的文件内容
f=codecs.open("D:/fenci_shuju/shuju.txt", 'r',encoding="utf8")

target = codecs.open("D:/fenci_shuju/zheng_li.txt", 'w',encoding="utf8")

print('open files')
line_num=1
line = f.readline()

#循环遍历每一行，并对这一行进行分词操作
#如果下一行没有内容的话，就会readline会返回-1，则while -1就会跳出循环
while line:
    print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()

#关闭两个文件流，并退出程序
f.close()
target.close()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Word2Vec第一个参数代表要训练的语料
    # sg=1 表示使用Skip-Gram模型进行训练
    # size 表示特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # window 表示当前词与预测词在一个句子中的最大距离是多少
    # min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # workers 表示训练的并行数
    #sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)

def A():
    #首先打开需要训练的文本
    shuju = open('D:/fenci_shuju/zheng_li.txt', 'rb')
    #通过Word2vec进行训练
    model = Word2Vec(LineSentence(shuju), sg=1, window=10, min_count=5, workers=15,sample=1e-3)
    #保存训练好的模型
    model.wv.save_word2vec_format('D:/fenci_shuju/jiudian_pinglun_Test.vector')

    print('训练完成')
    Excel = open("baoGao.csv", 'w', newline = '')   #打开表格文件，若表格文件不存在则创建
    writ = csv.writer(Excel)    #创建一个csv的writer对象用于写每一行内容
    
    writ.writerow(model.wv.most_similar(['服务'],topn=20))
    writ.writerow(model.wv.most_similar(['位置'],topn=20))
    writ.writerow(model.wv.most_similar(['设施'],topn=20))
    writ.writerow(model.wv.most_similar(['卫生'],topn=20))
    writ.writerow(model.wv.most_similar(['性价比'],topn=20))

if __name__ == '__main__':
    A()
    

time_end=time.time()
print("time:",time_end-time_begin)