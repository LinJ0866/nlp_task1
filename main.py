import os
import shutil
import pynlpir
import thulac
import pkuseg


pynlpir.open()
thul = thulac.thulac()
pku = pkuseg.pkuseg() 

# 任务底层接口，封装文件操作等底层API
class frame:
    # 需确认提示，用户确认则返回True，否则False
    def modal(self, info):
        x = input(info+'\n如继续请按【Y/y】，并回车确认\n')
        if x == 'Y' or x == 'y':
            return True
        return False

    #遍历文件夹，返回字符串切片
    def listDir(self, dirPath):
        dirList = os.listdir(dirPath)
        if len(dirList) == 0:
            return []

        if dirList[0].isdigit():
            dirList.sort(key=lambda x:int(x))
        
        return dirList
    
    # 写文件
    # 支持切片换行（最后一行不换行）、对象等数据结构的写入
    # 以utf-8编码保存
    def writeFile(self, filePath, content):
        if type(content).__name__ == 'list':
            with open(filePath, 'w+', encoding='utf-8') as f:
                f.write(str(content[0]))
                for i in range(len(content)-1):
                    f.write(','+str(content[i+1]))
            return
        print(type(content))

    # 打开GB18030编码的文本文件
    def readGB18030File(self, filePath):
        str = ''
        with open(filePath, 'r', encoding='GB18030', errors='ignore') as f:
            for i in f.readlines():
                str += i
        return str
    
    # 打开utf-8编码的文本文件
    def readUtf8File(self, filePath):
        str = ''
        with open(filePath, 'r', encoding='utf-8') as f:
            for i in f.readlines():
                str += i
        return str

# 继承自任务类，提供NLP任务公共部分接口
class nlpTask(frame):
    def __init__(self, datasetDir, isClassify):
        # 词性筛选表，若choose为空，则采用黑名单
        # NLPIR
        self.NLPIRChoose = [] #['noun', 'verb', 'adjective', 'adverb']
        self.NLPIRIgnore = [None, 'preposition', 'conjunction', 'particle', '', 'punctuation mark']

        # 是否已加载处理完成数据
        self.isProcessed = False

        self.preProcessingDir = os.path.join('processing','preProcessing')
        self.datasetDir = datasetDir
        self.dirList = self.listDir(self.datasetDir)
        
        if isClassify:
            self.readLabels()
    
    # 读取分类标签
    # 读取根目录的labels.txt
    # 若根目录labels.txt文件不存在，则利用文件夹名新建labels.txt（createNewLabelsFile）；否则返回labels数组
    def readLabels(self):
        if not(os.path.exists('labels.txt')):
            print("labels.txt文件不存在，已自动创建")
            self.createNewLabelsFile()
            self.labels = self.dirList
            return

        labels = self.readUtf8File('labels.txt')
        self.labels = labels.split(',')
        if len(self.labels) != len(self.dirList):
            print("【错误】labels文件label个数与数据集不符，请检查")
            exit(1)

    # 创建数据集
    def createNewLabelsFile(self):
        self.writeFile('labels.txt', self.dirList)

    # 中科院分词
    def NLPIRSeg(self, article):
        return pynlpir.segment(article)
    
    # 清华分词
    def thulSeg(self, article):
        return thul.cut(article)

    # 北大分词
    def pkuSeg(self, article):
        return pku.cut(article)

    # jieba分词

    # 预处理
    # 中文分词+虚词过滤
    # 参数说明
    # mode: 使用分词工具的类型，可选参数为['NLPIR', 'thul', 'jieba']
    def preProcessing(self, mode):
        if os.path.exists(self.preProcessingDir):
            if self.modal('【警告】此操作将删除processing/preProcessing文件夹内所有内容。'):
                shutil.rmtree(self.preProcessingDir)
            else:
                print('用户已取消操作')
                exit(1)
            
        for dir in self.dirList:
            newDir = os.path.join(self.datasetDir, dir)
            newCreateDir = os.path.join(self.preProcessingDir, dir)
            os.makedirs(newCreateDir)

            for file in self.listDir(newDir):
                fileContent = self.readGB18030File(os.path.join(newDir, file))
                saveWords = []
                if mode == 'NLPIR':
                    splitWord = self.NLPIRSeg(fileContent)
                    for word in splitWord:
                        if len(self.NLPIRChoose) != 0:
                            if word[1] in self.NLPIRChoose:
                                saveWords.append(word[0].replace('\n', ''))
                        else:
                            if word[1] not in self.NLPIRIgnore:
                                saveWords.append(word[0].replace('\n', ''))
                elif mode == 'thul':
                    pass
                elif mode == 'jieba':
                    pass
                else:
                    print('输入有误，分词失败')
                    exit(1)
                self.writeFile(os.path.join(newCreateDir, file), saveWords)
            

# 任务1.1 文本分类系统
class classifyNLP(nlpTask):
    def process(self):
        pass

# 任务1.2 信息检索系统
class queryNLP(nlpTask):
    def process(self):
        pass

if __name__ == '__main__':
    classifyTask = classifyNLP('./data/train', True)

    # print(classifyTask.thulSeg(classifyTask.readGB18030File('./data/train/1/1 (1).txt')))
    # print(classifyTask.pkuSeg(classifyTask.readGB18030File('./data/train/1/1 (1).txt')))
    
    classifyTask.preProcessing('NLPIR')
    classifyTask.process()
