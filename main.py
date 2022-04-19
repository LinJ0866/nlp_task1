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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier


pynlpir.open()
thul = thulac.thulac()
pku = pkuseg.pkuseg()
jieba.enable_paddle()

# 任务底层接口，封装文件操作等底层API
class frame:
    # 需确认提示，用户确认则返回True，否则False
    def modal(self, info):
        print()
        x = input(info+'\n如继续请按【Y/y】，并回车确认\n')
        if x == 'Y' or x == 'y':
            return True
        return False

    # 封装遍历文件夹排序
    def sortDirList(self, dirList):
        dirList.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
        return dirList
    

    #遍历文件夹，返回字符串切片
    def listDir(self, dirPath):
        dirList = os.listdir(dirPath)
        if len(dirList) == 0:
            return []

        return self.sortDirList(dirList)
    
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

# 任务1 文本分类系统（with 检索数据集）
class classifyNLP(frame):
    # 参数说明
    # ratio：表示测试集比例，若testPath为None，将划分训练集
    # ramdonState：用于划分数据集，None表示纯随机
    # processingPath：处理中间文件存放处
    # mode: 分词工具,使用分词工具的类型，可选参数为['NLPIR', 'thul', 'pku', 'jieba']
    # featureSelectionMode：使用特征选择的方法，可选参数为['IG', 'MI', 'CHI']
    # featureCalMode：使用特征权重计算的方法，可选参数为['TF-IDF', 'TF', 'IDF']
    # modelMode：使用分类器的方法，可选参数为['GaussianNB', 'tree', 'MLP']
    def __init__(self, trainPath, testPath, testRatio=0.3, randomState=None, processingPath='./processing', mode='NLPIR', featureSelectionMode='IG', featureCalMode="TF-IDF", modelMode='GaussianNB'):
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
        self.jiebaIgnore = [None,'c','w','p','u','e','o','y']

        self.trainPath = trainPath
        self.testPath = testPath
        self.ratio = testRatio
        self.randomState = randomState # 用于划分数据集，None表示纯随机
        self.processingPath = processingPath
        self.mode = mode # mode: 使用分词工具的类型，可选参数为['NLPIR', 'thul', 'pku', 'jieba']
        self.featureSelectionMode = featureSelectionMode # mode：使用特征选择的方法，可选参数为['IG', 'MI', 'CHI']
        self.featureCalMode = featureCalMode # mode:使用特征权重计算的方法，可选参数为['TF-IDF', 'TF', 'IDF']
        self.modelMode = modelMode

        self.labels = []
        # 训练集数据集词表
        self.wordsMap = {
            'tfSum': 0,     # 总共出现的词频
            'docSum': 0,
            'count': [],          # 各分类的数量
            'words': [],          # 词向量
        }
        # 测试集数据集词表
        self.wordsMap1 = {
            'tfSum': 0,     # 总共出现的词频
            'docSum': 0,
            'count': [],          # 各分类的数量
            'words': [],          # 词向量
        }
        self.trainDatasetPath = []
        self.trainDatasetLabel = []
        self.testDatasetPath = []
        self.testDatasetLabel = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.features = [] # 存储feature的词表id
        self.featureText = []


        self.readLabels()
        self.readDataset()
        
    
    # 读取分类标签
    # 读取根目录的labels.txt
    # 若根目录labels.txt文件不存在，则利用文件夹名新建labels.txt（createNewLabelsFile）；否则返回labels数组
    def readLabels(self):
        self.labels = self.listDir(self.trainPath)

        if not(os.path.exists('labels.txt')):
            self.createNewLabelsFile()
            return

        labels = self.readUtf8File('labels.txt')
        labels = labels.split(',')
        if len(labels) != len(self.labels):
            print("【错误】labels文件label个数与数据集不符，请检查")
            exit(1)
        self.labels = labels

    # 创建数据集标签
    def createNewLabelsFile(self):
        print("labels.txt文件不存在，已自动创建")
        self.writeFile('labels.txt', self.labels)

    # 读入数据集
    def readDataset(self):
        datasetFile = os.path.join(self.processingPath, 'datasets.json')
        if not(os.path.exists(datasetFile)):
            self.createNewDataset()
            return
        
        datasets = self.readUtf8File(datasetFile)
        self.trainDatasetPath = datasets['train']['path']
        self.trainDatasetLabel = datasets['train']['label']
        self.testDatasetPath = datasets['test']['path']
        self.testDatasetLabel = datasets['test']['label']
        print('数据集读入成功')
        
    # 创建数据集
    # 若未提供测试集文件夹，则进行拆分
    def createNewDataset(self):
        trainDir = self.listDir(self.trainPath)
        for i in range(len(trainDir)):
            newPath = os.path.join(self.trainPath, trainDir[i])
            for file in self.listDir(newPath):
                self.trainDatasetPath.append(os.path.join(newPath, file))
                self.trainDatasetLabel.append(i)
        
        if self.testPath == None:
            self.trainDatasetPath, self.testDatasetPath, self.trainDatasetLabel, self.testDatasetLabel = train_test_split(self.trainDatasetPath, self.trainDatasetLabel, test_size=self.ratio, random_state=self.randomState)
        else:
            testDir = self.listDir(self.testPath)
            for i in range(len(testDir)):
                newPath = os.path.join(self.testPath, testDir[i])
                for file in self.listDir(newPath):
                    self.testDatasetPath.append(os.path.join(newPath, file))
                    self.testDatasetLabel.append(i)
        
        datasetJson = {
            'train': {
                "path": self.trainDatasetPath,
                "label": self.trainDatasetLabel
            },
            "test": {
                "path": self.testDatasetPath,
                "label": self.testDatasetLabel
            }
        }

        self.writeFile(os.path.join(self.processingPath, 'datasets.json'), datasetJson)
        print('数据集创建成功')


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
    def preProcessing(self):
            wordMapFilePath = os.path.join(self.processingPath, 'wordsMap.json')
            wordMapTestFilePath = os.path.join(self.processingPath, 'wordsMapTest.json')
            
            if os.path.exists(wordMapFilePath) and\
                    not(self.modal('【警告】模型已加载训练集预处理词表数据，此操作可能会删除wordsMap.json和dataset/train文件夹内的内容。')):
                self.wordsMap = self.readUtf8File(wordMapFilePath)
                print('----train预处理词表加载完成----')
            else:
                print('---即将开始预处理train相关数据----')
                datasetPath = os.path.join(self.processingPath, 'dataset', 'train')
                if os.path.exists(datasetPath):
                    shutil.rmtree(datasetPath)
                self.splitAndSave('train')

            if os.path.exists(wordMapTestFilePath) and\
                    not(self.modal('【警告】模型已加载测试集集预处理词表数据，此操作可能会删除wordsMap.json和dataset/test文件夹内的内容。')):
                self.wordsMap1 = self.readUtf8File(wordMapTestFilePath)
                print('----test预处理词表加载完成----')
            else:
                print('---即将开始预处理test相关数据----')
                datasetPath = os.path.join(self.processingPath, 'dataset', 'test')
                if os.path.exists(datasetPath):
                    shutil.rmtree(datasetPath)
                self.splitAndSave('test')

    # 中文分词+虚词过滤，顺便获取词频(TF)与词文档频(DF)
    # mode: ['train', 'test']  训练集、测试集分开处理
    def splitAndSave(self, mode):
        fileList = []
        labels = self.labels
        labelIds = []
        ruleChoose = []
        ruleIgnore = []
        splitWord = []
        wordsMap = {}

        saveDir = os.path.join(self.processingPath, 'dataset', mode)
        wordMapFilePath = ""
        os.makedirs(saveDir, exist_ok=True)

        if mode == 'test':
            fileList = self.testDatasetPath
            labelIds = self.testDatasetLabel
            wordMapFilePath = os.path.join(self.processingPath, 'wordsMapTest.json')
            wordsMap = self.wordsMap1
        elif mode == 'train':
            fileList = self.trainDatasetPath
            labelIds = self.trainDatasetLabel
            wordMapFilePath = os.path.join(self.processingPath, 'wordsMap.json')
            wordsMap = self.wordsMap
        else:
            print('参数错误：仅可输入 train 或 test'.format(mode))
            exit(1)
        
        # 初始化
        for label in labels:
            wordsMap['count'].append(0)
            wordsMap[label] = {
                "tf": [],
                "df": [] 
            }

        for (id, filePath) in enumerate(tqdm(fileList)):
            fileContent = self.readGB18030File(filePath)
            fileContent = self.filtering(fileContent)
            labelId = labelIds[id]
            label = labels[labelId]
            saveWords = {}

            # 分词
            if self.mode == 'NLPIR':
                ruleChoose = self.NLPIRChoose
                ruleIgnore = self.NLPIRIgnore
                splitWord = self.NLPIRSeg(fileContent)
            elif self.mode == 'thul':
                ruleChoose = self.thulChoose
                ruleIgnore = self.thulIgnore
                splitWord = self.thulSeg(fileContent)
            elif self.mode == 'pku':
                ruleChoose = self.pkuChoose
                ruleIgnore = self.pkuIgnore
                splitWord = self.pkuSeg(fileContent)
            elif self.mode == 'jieba':
                ruleChoose = self.jiebaChoose
                ruleIgnore = self.jiebaIgnore
                splitWord = self.jiebaSeg(fileContent)
            else:
                print('参数错误：暂不支持{}，分词失败'.format(self.mode))
                exit(1)
            
            # 按词性进行过滤
            for word in splitWord:
                # 若len(ruleChoose)不为0则使用白名单模式，否则黑名单模式
                if (len(ruleChoose) != 0 and word[1] in ruleChoose) or \
                        (len(ruleChoose) == 0 and word[1] not in ruleIgnore):
                    
                    # 查看词表中是否出现过该单词
                    p = wordsMap['words'].index(word[0]) if word[0] in wordsMap['words'] else -1
                    if p == -1:
                        wordsMap['words'].append(word[0])
                        p = len(wordsMap['words'])-1
                    
                    # 补齐wordsMap[label][count]，[exist]列表至p处
                    for i in range(max(0, p+1-len(wordsMap[label]['tf']))):
                        wordsMap[label]['tf'].append(0)
                        wordsMap[label]['df'].append(0)
                    
                    if saveWords.get(word[0]) == None:
                        wordsMap[label]['df'][p] += 1
                        saveWords[word[0]] = 1
                    else:
                        saveWords[word[0]] += 1
                    
                    wordsMap['tfSum'] += 1
                    wordsMap[label]['tf'][p] += 1
            
            self.writeFile(os.path.join(saveDir, str(id)+'.json'), saveWords)
            wordsMap['count'][labelId] += 1
        
        wordsMap['docSum'] = sum(wordsMap['count'])

        if mode == 'train':
            self.wordsMap = wordsMap
        else:
            self.wordsMap1 = wordsMap

        self.writeFile(wordMapFilePath, wordsMap)
    
    # 正文提取相关规则
    def filtering(self, content):
        content = content.replace(' ', '')
        content = content.replace('\n', '')
        return content

    def wordInCatagory(self, type1, catagory, i, mode):
        if mode == 'train':
            wordsMap = self.wordsMap
        else:
            wordsMap = self.wordsMap1
        if i >= len(wordsMap[catagory][type1]):
            return 0
        return wordsMap[catagory][type1][i]
    
    def featureSelection(self):
        if self.featureSelectionMode == 'IG':
            self.featureSelectionIG()
        elif self.featureSelectionMode == 'MI':
            self.featureSelectionMI()
        elif self.featureSelectionMode == 'CHI':
            self.featureSelectionCHI()


    # 信息增益
    def featureSelectionIG(self):
        featurePath = os.path.join(self.processingPath, 'features.json')
        featureIGPath = os.path.join(self.processingPath, 'featureIG.json')
        # 加载历史文件
        if os.path.exists(featurePath) and\
                not(self.modal('【警告】模型已加载特征选择记录，此操作可能会删除features.json和featureChoose.json的内容。')):
            feature = self.readUtf8File(featurePath)
            for i in range(len(feature)):
                self.features.append(feature[str(i)][0])
                self.featureText.append(self.wordsMap['words'][feature[str(i)][0]])

            print('----特征加载完成----')
            return
        
        labelLen = len(self.labels)
        docSum = self.wordsMap['docSum'] # 全部文档数

        igDict = {}

        # 计算sum(-p(cj)log(p(cj)))
        EntPcs = []
        EntPc = 0
        for i in range(labelLen):
            EntPcs.append(self.wordsMap['count'][i]/docSum)
            EntPc -= EntPcs[i]*math.log2(EntPcs[i])


        print('----开始计算特征信息增益----')
        for i in tqdm(range(len(self.wordsMap['words']))):
            EntPFeature = EntPc
            countSumFeature = 0
            EntFeatureT = 0
            EntFeatureF = 0

            for label in self.labels:
                countSumFeature += self.wordInCatagory('df', label, i, 'train') # 含北京文档数
            
            for label in self.labels:
                # 采取拉普拉斯平滑
                pt = (self.wordInCatagory('df', label, i, 'train')+1)/(countSumFeature+labelLen)
                pf = (docSum-self.wordInCatagory('df', label, i, 'train')+1)/(docSum-countSumFeature+labelLen)

                EntFeatureT += pt*math.log2(pt)
                EntFeatureF += pf*math.log2(pf)
            
            pFeature = countSumFeature/docSum
            EntPFeature += pFeature*EntFeatureT
            EntPFeature += (1-pFeature)*EntFeatureT

            igDict[self.wordsMap['words'][i]] = EntPFeature
            # print(self.wordsMap['words'][i], EntPFeature)
        
        self.writeFile(featureIGPath, igDict)
        igList = sorted(igDict.items(), key=lambda d:d[1], reverse=True) 

        igJson = {}
        for i in range(1000):
            p = self.wordsMap['words'].index(igList[i][0])
            igJson[i] = [p, igList[i][0], igList[i][1]]
            self.features.append(p)
            self.featureText.append(igList[i][0])
        self.writeFile(featurePath, igJson)
        print('----特征选取完成----')

    def featureSelectionMI(self):
        featurePath = os.path.join(self.processingPath, 'features.json')
        featureIGPath = os.path.join(self.processingPath, 'featureMI.json')
        # 加载历史文件
        if os.path.exists(featurePath) and\
                not(self.modal('【警告】模型已加载特征选择记录，此操作可能会删除features.json和featureMI.json的内容。')):
            feature = self.readUtf8File(featurePath)
            for i in range(len(feature)):
                self.features.append(feature[str(i)][0])
                self.featureText.append(self.wordsMap['words'][feature[str(i)][0]])

            print('----特征加载完成----')
            return
        
        labelLen = len(self.labels)
        docSum = self.wordsMap['docSum'] # 全部文档数

        igDict = {}

        print('----开始计算特征互信息----')
        for i in tqdm(range(len(self.wordsMap['words']))):
            countSumFeature = 0
            MI = 0

            for label in self.labels:
                countSumFeature += self.wordInCatagory('df', label, i, 'train') # 含北京文档数
            
            for (labelId, label) in enumerate(self.labels):
                MI += (self.wordsMap['count'][labelId]+1)*math.log2((self.wordInCatagory('df', label, i, 'train')+1)/(self.wordsMap['count'][labelId]+1)/(countSumFeature+labelLen)*(docSum++labelLen))

            igDict[self.wordsMap['words'][i]] = MI
        
        self.writeFile(featureIGPath, igDict)
        igList = sorted(igDict.items(), key=lambda d:d[1]) 

        igJson = {}
        for i in range(1000):
            p = self.wordsMap['words'].index(igList[i][0])
            igJson[i] = [p, igList[i][0], igList[i][1]]
            self.features.append(p)
            self.featureText.append(igList[i][0])
        self.writeFile(featurePath, igJson)
        print('----特征选取完成----')

    # 信息增益
    def featureSelectionCHI(self):
        featurePath = os.path.join(self.processingPath, 'features.json')
        featureIGPath = os.path.join(self.processingPath, 'featureCHI.json')
        # 加载历史文件
        if os.path.exists(featurePath) and\
                not(self.modal('【警告】模型已加载特征选择记录，此操作可能会删除features.json和featureChoose.json的内容。')):
            feature = self.readUtf8File(featurePath)
            for i in range(len(feature)):
                self.features.append(feature[str(i)][0])
                self.featureText.append(self.wordsMap['words'][feature[str(i)][0]])

            print('----特征加载完成----')
            return
        
        labelLen = len(self.labels)
        docSum = self.wordsMap['docSum'] # 全部文档数

        igDict = {}

        # 计算sum(-p(cj)log(p(cj)))
        EntPcs = []
        EntPc = 0
        for i in range(labelLen):
            EntPcs.append(self.wordsMap['count'][i]/docSum)
            EntPc -= EntPcs[i]*math.log2(EntPcs[i])


        print('----开始计算特征卡方----')
        for i in tqdm(range(len(self.wordsMap['words']))):
            EntPFeature = EntPc
            countSumFeature = 0
            ChiFeature = 0

            for label in self.labels:
                countSumFeature += self.wordInCatagory('df', label, i, 'train') # T
            
            for (labelId, label) in enumerate(self.labels):
                # 采取拉普拉斯平滑
                A = self.wordInCatagory('df', label, i, 'train')
                B = countSumFeature - A
                C = self.wordsMap['count'][labelId] - A
                D = docSum - A - B - C

                ChiFeature += (A*D-C*B)*(A*D-C*B)/((A+C)*(B+D)*(A+B)*(C+D))*(self.wordsMap['count'][labelId]/docSum)
            
            igDict[self.wordsMap['words'][i]] = ChiFeature
        
        self.writeFile(featureIGPath, igDict)
        igList = sorted(igDict.items(), key=lambda d:d[1], reverse=True) 

        igJson = {}
        for i in range(1000):
            p = self.wordsMap['words'].index(igList[i][0])
            igJson[i] = [p, igList[i][0], igList[i][1]]
            self.features.append(p)
            self.featureText.append(igList[i][0])
        self.writeFile(featurePath, igJson)
        print('----特征选取完成----')

    def featureCal(self, mode):
        fileFeatures = []
        datasetPath = os.path.join(self.processingPath, 'dataset', mode)
        if mode == 'train':
            featuresPath = os.path.join(self.processingPath, 'trainFeatures.json')
            dirList = self.trainDatasetPath
            labelList = self.trainDatasetLabel
            wordsMap = self.wordsMap
        else:
            featuresPath = os.path.join(self.processingPath, 'testFeatures.json')
            dirList = self.testDatasetPath
            labelList = self.testDatasetLabel
            wordsMap = self.wordsMap1

        # 加载历史文件
        if os.path.exists(featuresPath) and\
                not(self.modal('【警告】模型已完成特征权重计算，此操作可能会删除{}的内容。'.format(featuresPath))):
            fileFeatures = self.readUtf8File(featuresPath)
            # 初始化个模板
            featureBinaryT = []
            for i in range(len(self.features)):
                featureBinaryT.append(0)
            
            for fileInfo in fileFeatures:
                featureBinary = featureBinaryT.copy()

                for i in fileInfo['features'].keys():
                    featureBinary[int(i)] = fileInfo['features'][i]
                
                if mode == 'train':
                    self.X_train.append(featureBinary)
                    self.y_train.append(fileInfo['labels'])
                else:
                    self.X_test.append(featureBinary)
                    self.y_test.append(fileInfo['labels'])
            
            if mode == 'train':
                self.X_train = pd.DataFrame(self.X_train,
                                index=range(len(fileFeatures)),
                                columns=self.featureText)
            else:
                self.X_test = pd.DataFrame(self.X_test,
                                index=range(len(fileFeatures)),
                                columns=self.featureText)
            
            print('----{}数据集加载完成----'.format(mode))
            return

        print('-----特征权重计算开始-----')
        docSum = wordsMap['docSum']

        for (i, file) in enumerate(tqdm(dirList)):
                # fileInfo用于存储json，节省空间为主，featureBinary用于分类器，方便为主
                fileInfo = {
                    'id': i,
                    'path': file,
                    'features': {},
                    'labels': labelList[i]
                }
                featureBinary = []
                fileJson = self.readUtf8File(os.path.join(datasetPath, str(i)+'.json'))
                for feature in self.features:
                    feature = int(feature)

                    if fileJson.get(self.wordsMap["words"][feature]) == None:
                        featureBinary.append(0)
                    else:
                        tf = fileJson.get(self.wordsMap["words"][feature])
                        df = 0
                        for label in self.labels:
                            df += self.wordInCatagory('df', label, feature, mode)
                        if self.featureCalMode == 'TF-IDF':
                            w = tf*math.log2(docSum/(df+1))
                        elif self.featureCalMode == 'TF':
                            w = tf
                        elif self.featureCalMode == 'IDF':
                            w = math.log2(docSum/df)
                        
                        fileInfo['features'][self.features.index(feature)] = w
                        featureBinary.append(w)
                
                

                if mode == 'train':
                    self.X_train.append(featureBinary)
                    self.y_train.append(labelList[i])
                else:
                    self.X_test.append(featureBinary)
                    self.y_test.append(labelList[i])
                fileFeatures.append(fileInfo)    
        if mode == 'train':
            self.X_train = pd.DataFrame(self.X_train,
                                    index=range(len(fileFeatures)),
                                    columns=self.featureText)
        else:
            self.X_test = pd.DataFrame(self.X_test,
                                    index=range(len(fileFeatures)),
                                    columns=self.featureText)
            
        json.dump(fileFeatures, open(featuresPath, 'w+', encoding="utf-8"), indent=2, ensure_ascii=False)
        print('----特征计算完成---')
        
    def train(self):
        if self.modelMode == 'GaussianNB':
            self.trainGaussianNB()
        elif self.modelMode == 'tree':
            self.trainTree()
        elif self.modelMode == 'MLP':
            self.trainMLP()
    
    def trainGaussianNB(self):
        gnb = GaussianNB()
        self.y_pred = gnb.fit(self.X_train, self.y_train).predict(self.X_test)
    
    def trainTree(self):
        clf = tree.DecisionTreeClassifier()
        self.y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
    
    def trainMLP(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                hidden_layer_sizes=(20, 10), random_state=1)
        self.y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)

    def calError(self):
        micro = [0, 0, 0, 0]
        macro = [0, 0, 0]
        err = 0

        featuresPath = os.path.join(self.processingPath, 'testFeatures.json')
        fileFeatures = self.readUtf8File(featuresPath)
        errorInfo = []
        errorInfo.append('{}：{}，{}'.format('文件'.ljust(60), '预测', '标签'))
        for (i, (index, _)) in enumerate(self.X_test.iterrows()):
            if self.y_test[i] != self.y_pred[i]:
                err += 1
                label = self.labels[int(fileFeatures[index]['labels'])]
                errorInfo.append('{}：{}，{}'.format(fileFeatures[index]['path'].ljust(60), self.labels[self.y_pred[i]], label))
            
        self.writeFile('result.txt', '\n'.join(errorInfo))

        print('[T] AP is {}'.format(accuracy_score(self.y_test, self.y_pred)))
        print('[T] MicroP is {}, MicroR is {}, MicroF1 is {}'.format(precision_score(self.y_test, self.y_pred, average='micro'), recall_score(self.y_test, self.y_pred, average='micro'), f1_score(self.y_test, self.y_pred, average='micro')))
        print('[T] MacroP is {}, MacroR is {}, MacroF1 is {}'.format(precision_score(self.y_test, self.y_pred, average='macro'), recall_score(self.y_test, self.y_pred, average='macro'), f1_score(self.y_test, self.y_pred, average='macro')))
        
        # AP
        testLen = self.X_test.shape[0]
        print("AP out of a total %d is : %f"
            % (testLen, (testLen-err)/testLen))
                
        for label in range(len(self.labels)):
            marix = [0, 0, 0, 0]
            for i in range(len(self.y_pred)):
                if(self.y_test[i] == label and self.y_pred[i] == label):
                    marix[0] += 1
                elif(self.y_test[i] == label and self.y_pred[i] != label):
                    marix[1] += 1
                elif(self.y_test[i] != label and self.y_pred[i] == label):
                    marix[2] += 1
                elif(self.y_test[i] != label and self.y_pred[i] != label):
                    marix[3] += 1

            for i in range(4):
                micro[i] += marix[i]
            
            p = marix[0]/(marix[0]+marix[2])
            r = marix[0]/(marix[0]+marix[1])

            macro[0] += p
            macro[1] += r
            macro[2] += 2*p*r/(p+r)


        for i in range(3):
            macro[i] /= len(self.labels)
        for i in range(4):
            micro[i] /= len(self.labels)
        p = micro[0]/(micro[0]+micro[2])
        r = micro[0]/(micro[0]+micro[1])
        print('MicroP is {}, MicroR is {}, MicroF1 is {}'.format(p, r, 2*p*r/(p+r)))
        print('MacroP is {}, MacroR is {}, MacroF1 is {}'.format(macro[0], macro[1], macro[2]))
    

    # 参数说明
    # testRatio：表示测试集比例，若testPath为None，将划分训练集
    # ramdonState：用于划分数据集，None表示纯随机
    # processingPath：处理中间文件存放处
    # mode: 分词工具,使用分词工具的类型，可选参数为['NLPIR', 'thul', 'pku', 'jieba']
    # featureSelectionMode：使用特征选择的方法，可选参数为['IG', 'MI', 'CHI']
    # featureCalMode：使用特征权重计算的方法，可选参数为['TF-IDF', 'TF', 'IDF']
    # modelMode：使用分类器的方法，可选参数为['GaussianNB', 'tree', 'MLP']
if __name__ == '__main__':
    classifyTask = classifyNLP('./data/train', '/data/test', mode="NLPIR", featureSelectionMode='IG', featureCalMode="TF-IDF", modelMode="MLP")
    
    classifyTask.preProcessing()
    classifyTask.featureSelection()
    classifyTask.featureCal('train')
    classifyTask.featureCal('test')
    classifyTask.train()
    classifyTask.calError()
