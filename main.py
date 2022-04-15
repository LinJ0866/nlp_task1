import os
import shutil
import json

import pynlpir
import thulac
import pkuseg
import jieba
import jieba.posseg as pseg

from tqdm import tqdm


pynlpir.open()
thul = thulac.thulac()
pku = pkuseg.pkuseg()
jieba.enable_paddle()

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
        typeName = type(content).__name__
        if typeName == 'list':
            with open(filePath, 'w+', encoding='utf-8') as f:
                f.write(str(content[0]))
                for i in range(len(content)-1):
                    f.write(','+str(content[i+1]))
            return
        elif typeName == 'dict':
            json.dump(content, open(filePath, 'w+', encoding='utf-8'), indent=2, ensure_ascii=False)
            return
        
        print(typeName)

    # 打开GB18030编码的文本文件
    def readGB18030File(self, filePath):
        str = ''
        with open(filePath, 'r', encoding='GB18030', errors='ignore') as f:
            for i in f.readlines():
                # Todo： 正文提取
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
        # thu
        self.thulChoose = []
        self.thulIgnore = [None,'c','w','p','u','e','o','y']
        # pku
        self.pkuChoose = []
        self.pkuIgnore = [None,'c','w','p','u','e','o','y']
        # jieba-paddle
        self.jiebaChoose = []
        self.jiebaIgnore = []

        # 数据集词表
        self.wordsMap = {
            'words': [],
            'globalWords': []
        }

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
    def jiebaSeg(self, article):
        wordSeg = []
        words = pseg.cut(article,use_paddle=True)
        for word, flag in words:
            wordSeg.append((word, flag))
        return wordSeg

    # 预处理
    # 中文分词+虚词过滤
    # 参数说明
    # mode: 使用分词工具的类型，可选参数为['NLPIR', 'thul', 'pku', 'jieba']
    # needSplitResult：是否需要输出分词结果，默认值False
    def preProcessing(self, mode, needSplitResult=False):
        if needSplitResult and os.path.exists(self.preProcessingDir):
            if self.modal('【警告】此操作将删除processing/preProcessing文件夹内所有内容。'):
                shutil.rmtree(self.preProcessingDir)
            else:
                print('用户已取消操作')
                exit(1)

        print('-----开始进行预处理-----')
        categoryCount = len(self.dirList)
        for (catagoryId, dir) in enumerate(self.dirList):
            label = self.labels[catagoryId]
            self.wordsMap[label] = []

            newDir = os.path.join(self.datasetDir, dir)
            newCreateDir = os.path.join(self.preProcessingDir, dir)
            if needSplitResult:
                os.makedirs(newCreateDir)

            filesInDir = self.listDir(newDir)
            pbar = tqdm(filesInDir)

            for (fileId, _) in enumerate(pbar):
                pbar.set_description(f"{catagoryId+1}/{categoryCount}")

                file = filesInDir[fileId]
                fileContent = self.readGB18030File(os.path.join(newDir, file))

                # 正则处理空格与换行
                fileContent = fileContent.replace(' ', '')
                fileContent = fileContent.replace('\n', '')

                ruleChoose = []
                ruleIgnore = []
                splitWord = []
                saveWords = []

                # 分词
                if mode == 'NLPIR':
                    ruleChoose = self.NLPIRChoose
                    ruleIgnore = self.NLPIRIgnore
                    splitWord = self.NLPIRSeg(fileContent)
                elif mode == 'thul':
                    ruleChoose = self.thulChoose
                    ruleIgnore = self.thulIgnore
                    splitWord = self.thulSeg(fileContent)
                elif mode == 'pku':
                    ruleChoose = self.pkuChoose
                    ruleIgnore = self.pkuIgnore
                    splitWord = self.pkuSeg(fileContent)
                elif mode == 'jieba':
                    ruleChoose = self.jiebaChoose
                    ruleIgnore = self.jiebaIgnore
                    splitWord = self.jiebaSeg(fileContent)
                else:
                    print('参数错误：暂不支持{}，分词失败'.format(mode))
                    exit(1)
                
                # 按词性进行过滤
                for word in splitWord:
                    # 若len(ruleChoose)不为0则使用白名单模式，否则黑名单模式
                    if (len(ruleChoose) != 0 and word[1] in ruleChoose) or \
                            (len(ruleChoose) == 0 and word[1] not in ruleIgnore):
                        # 若需输出中间结果
                        if needSplitResult:
                            saveWords.append(word[0])
                        
                        # 查看词表中是否出现过该单词
                        p = self.wordsMap['words'].index(word[0]) if word[0] in self.wordsMap['words'] else -1
                        if p == -1:
                            self.wordsMap['words'].append(word[0])
                            self.wordsMap['globalWords'].append(0)
                            p = len(self.wordsMap['words'])-1
                        
                        # 补齐文章列表至p处
                        for i in range(max(0, p+1-len(self.wordsMap[label]))):
                            self.wordsMap[label].append(0)
                        
                        self.wordsMap[label][p] += 1
                        self.wordsMap['globalWords'][p] += 1
                
                if needSplitResult:
                    self.writeFile(os.path.join(newCreateDir, file), saveWords)

        self.writeFile('wordsMap.json', self.wordsMap)

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
    
    classifyTask.preProcessing('NLPIR', True)
    classifyTask.process()
