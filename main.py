from audioop import reverse
import os
import shutil
import json
import math
import re

import pynlpir
import thulac
import pkuseg
import jieba
import jieba.posseg as pseg

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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
        if typeName == 'str':
            with open(filePath, 'w+', encoding='utf-8') as f:
                f.writelines(content)
            return
        elif typeName == 'list':
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
            if filePath.find('.json') != -1:
                return json.load(f)
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
            'countGlobal': 0,     # 总共出现的词频
            'labels': [],         # 标签名称
            'count': [],          # 各分类的数量
            'words': [],          # 词向量
        }

        self.features = []

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
            self.wordsMap['labels'] = self.dirList
            return

        labels = self.readUtf8File('labels.txt')
        labels = labels.split(',')
        if len(labels) != len(self.dirList):
            print("【错误】labels文件label个数与数据集不符，请检查")
            exit(1)
        self.wordsMap['labels'] = labels

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
    def preProcessing(self, mode):
        if os.path.exists('processing/wordsMap.json') and\
                not(self.modal('【警告】模型已加载预处理词表数据，此操作可能会删除processing/wordsMap.json和processing/preProcessing文件夹内的内容。')):
            self.wordsMap = self.readUtf8File('processing/wordsMap.json')
            print('----预处理词表加载完成----')
            return
        
        print('-----开始进行预处理-----')
        shutil.rmtree(self.preProcessingDir)
        categoryCount = len(self.dirList)
        for (catagoryId, dir) in enumerate(self.dirList):
            label = self.wordsMap['labels'][catagoryId]
            self.wordsMap[label] = {
                'count': [],  # 单词在该分类中出现的频次
                'exist': []   # 单词在该分类文章中出现的文章数
            }

            # 生成中间输出
            newDir = os.path.join(self.datasetDir, dir)
            newCreateDir = os.path.join(self.preProcessingDir, dir)
            os.makedirs(newCreateDir)

            filesInDir = self.listDir(newDir)

            self.wordsMap['countGlobal'] += len(filesInDir)
            self.wordsMap['count'].append(len(filesInDir))

            # 进度条与处理文件夹名
            print(newDir)
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
                saveWords = {}

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
                                                
                        # 查看词表中是否出现过该单词
                        p = self.wordsMap['words'].index(word[0]) if word[0] in self.wordsMap['words'] else -1
                        if p == -1:
                            self.wordsMap['words'].append(word[0])
                            p = len(self.wordsMap['words'])-1
                        
                        # 补齐wordsMap[label][count]，[exist]列表至p处
                        for i in range(max(0, p+1-len(self.wordsMap[label]['count']))):
                            self.wordsMap[label]['count'].append(0)
                            self.wordsMap[label]['exist'].append(0)
                        
                        self.wordsMap[label]['count'][p] += 1
                        if saveWords.get(word[0]) == None:
                            self.wordsMap[label]['exist'][p] += 1
                            saveWords[word[0]] = 1
                        else:
                            saveWords[word[0]] += 1
                
                self.writeFile(os.path.join(newCreateDir, file.split('.')[-2]+'.json'), saveWords)

        self.writeFile('processing/wordsMap.json', self.wordsMap)

# 任务1.1 文本分类系统
class classifyNLP(nlpTask):
    def wordInCatagory(self, mode, catagory, i):
        if i >= len(self.wordsMap[catagory][mode]):
            return 0
        return self.wordsMap[catagory][mode][i]
    
    def featureSelectionIG(self):
        self.features = [] # 存储feature的词表id
        self.featureText = []
        # 加载历史文件
        if os.path.exists('processing/features.json') and\
                not(self.modal('【警告】模型已加载特征选择记录，此操作可能会删除processing/features.json和processing/featureChoose.json的内容。')):
            feature = self.readUtf8File('processing/features.json')
            for i in range(len(feature)):
                self.features.append(feature[str(i)][0])
                self.featureText.append(self.wordsMap['words'][feature[str(i)][0]])

            print('----特征加载完成----')
            return
        
        labelLen = len(self.wordsMap['labels'])
        countSum = sum(self.wordsMap['count']) # 全部文档数

        igDict = {}

        # 计算sum(-p(cj)log(p(cj)))
        EntPcs = []
        EntPc = 0
        for i in range(labelLen):
            EntPcs.append(self.wordsMap['count'][i]/countSum)
            EntPc -= EntPcs[i]*math.log2(EntPcs[i])


        print('----开始计算特征信息增益----')
        for i in tqdm(range(len(self.wordsMap['words']))):
            EntPFeature = EntPc
            countSumFeature = 0
            EntFeatureT = 0
            EntFeatureF = 0

            for label in self.wordsMap['labels']:
                countSumFeature += self.wordInCatagory('exist', label, i) # 含北京文档数
            
            for label in self.wordsMap['labels']:
                # 采取拉普拉斯平滑
                pt = (self.wordInCatagory('exist', label, i)+1)/(countSumFeature+labelLen)
                pf = (countSum-self.wordInCatagory('exist', label, i)+1)/(countSum-countSumFeature+labelLen)

                EntFeatureT += pt*math.log2(pt)
                EntFeatureF += pf*math.log2(pf)
            
            pFeature = countSumFeature/countSum
            EntPFeature += pFeature*EntFeatureT
            EntPFeature += (1-pFeature)*EntFeatureT

            igDict[self.wordsMap['words'][i]] = EntPFeature
            # print(self.wordsMap['words'][i], EntPFeature)
        
        self.writeFile('processing/featureChoose.json', igDict)
        igList = sorted(igDict.items(), key=lambda d:d[1], reverse=True) 

        igJson = {}
        for i in range(1000):
            p = self.wordsMap['words'].index(igList[i][0])
            igJson[i] = [p, igList[i][0], igList[i][1]]
            self.features.append(p)
            self.featureText.append(igList[i][0])
        self.writeFile('processing/features.json', igJson)
        print('----特征选取完成----')

    def featureCalTFIDF(self):
        fileFeatures = []
        self.dataset = [[], []]

        # 加载历史文件
        if os.path.exists('processing/train.json') and\
                not(self.modal('【警告】模型已完成特征权重计算，此操作可能会删除processing/train.json的内容。')):
            fileFeatures = self.readUtf8File('processing/train.json')
            # 初始化个模板
            featureBinaryT = []
            for i in range(len(self.features)):
                featureBinaryT.append(0)
            
            for fileInfo in fileFeatures:
                featureBinary = featureBinaryT.copy()

                for i in fileInfo['features'].keys():
                    featureBinary[int(i)] = fileInfo['features'][i]
                
                self.dataset[0].append(featureBinary)
                self.dataset[1].append(fileInfo['labels'])

            self.dataset[0] = pd.DataFrame(self.dataset[0],
                                index=range(len(fileFeatures)),
                                columns=self.featureText)
            # print(self.dataset[0])
            print('----数据集加载完成----')
            return

        print('-----特征权重计算开始-----')
        id = 0
        countSum = sum(self.wordsMap['count'])

        for (i, dir) in enumerate(tqdm(self.dirList)):
            for file in self.listDir(os.path.join(self.preProcessingDir, dir)):
                # fileInfo用于存储json，节省空间为主，featureBinary用于分类器，方便为主
                fileInfo = {
                    'id': id,
                    'path': os.path.join(self.preProcessingDir, dir, file),
                    'features': {},
                    'labels': i
                }
                featureBinary = []
                fileJson = self.readUtf8File(fileInfo['path'])
                for feature in self.features:
                    feature = int(feature)

                    if fileJson.get(self.wordsMap["words"][feature]) == None:
                        featureBinary.append(0)
                    else:
                        df = 0
                        for label in self.wordsMap['labels']:
                            df += self.wordInCatagory('exist', label, feature)
                        w = fileJson.get(self.wordsMap["words"][feature])*math.log2(countSum/df)
                        fileInfo['features'][self.features.index(feature)] = w
                        featureBinary.append(w)
                
                self.dataset[0].append(featureBinary)
                self.dataset[1].append(i)
                fileFeatures.append(fileInfo)    
                id += 1
        self.dataset[0] = pd.DataFrame(self.dataset[0],
                                index=range(len(fileFeatures)),
                                columns=self.featureText)
        json.dump(fileFeatures, open('processing/train.json', 'w+', encoding="utf-8"), indent=2, ensure_ascii=False)
        print('----特征计算完成---')
        
    def trainGaussianNB(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.dataset[0], self.dataset[1], test_size=0.3, random_state=0)
        # print(X_train)
        gnb = GaussianNB()
        self.y_pred = gnb.fit(X_train, y_train).predict(self.X_test)

    def calError(self):
        print("Number of mislabeled points out of a total %d points : %d"
            % (self.X_test.shape[0], (self.y_test != self.y_pred).sum()))

        fileFeatures = self.readUtf8File('processing/train.json')
        errorInfo = []
        errorInfo.append('{}：{}，{}'.format('文件'.ljust(60), '预测', '标签'))
        for (i, (index, _)) in enumerate(self.X_test.iterrows()):
            if self.y_test[i] != self.y_pred[i]:
                label = re.findall(r"\d+", fileFeatures[index]['path'])
                label = self.wordsMap['labels'][int(label[0])-1]

                errorInfo.append('{}：{}，{}'.format(fileFeatures[index]['path'].ljust(60), self.wordsMap['labels'][self.y_pred[i]], label))
        self.writeFile('result.txt', '\n'.join(errorInfo))


    def process(self):
        self.featureSelectionIG()
        self.featureCalTFIDF()
        self.trainGaussianNB()
        self.calError()

# 任务1.2 信息检索系统
class queryNLP(nlpTask):
    def process(self):
        pass

if __name__ == '__main__':
    classifyTask = classifyNLP('./data/train', True)
    
    classifyTask.preProcessing('NLPIR')
    classifyTask.process()
