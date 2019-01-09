
# -*- coding: utf-8 -*-
import torch
import jieba
# from seq2seq.evaluator import Predictor
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):

        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]

        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        aaa = []
        for tt in softmax_list[0]:
            for t in tt:
                # print(t)
                aaa.append(math.exp(t))
                # for t2 in t:
                #     aaa.append(math.exp(t2))
        aaa.sort()
        A = {}
        A['tgt_seq']=tgt_seq
        A['score']=aaa[::-1]
        return A

predictor1 = None
predictor2 = None

det = {"000900130001": "沐足店", "000900130009": "俱乐部", "000900130004": "棋牌室", "00090020": "村委牌坊", "000900050008": "其他",
     "000900050004": "独体楼", "00090004": "内街小巷", "0009000100060002": "篮球场", "000900130015": "美容店", "000900030002": "人行道",
     "000900130012": "会所", "000900030009": "街面", "000900050001": "物业小区", "000900020001": "实体银行", "000900060002": "城中村",
     "000900060011": "其他", "000900110007": "其他", "0009000100010014": "士多店", "000900130003": "按摩店", "000900130024": "其他",
     "0009000100010022": "菜市场", "000900110001": "开放式公园（含街心公园）", "0009000100010001": "超市", "0009000100020005": "饭店",
     "000900060004": "独体楼", "0009000100020015": "其他", "000900100012": "其他", "0009000100010015": "便利店",
     "0009000100030002": "宾馆", "0009000100010028": "其他", "00090023": "老人院", "000900070002": "火车站",
     "000900010010": "物流货运场所", "0009000100030006": "连锁酒店", "000900120005": "饭堂", "0009000100010018": "档口/商铺",
     "000900060008": "出租公寓", "000900130010": "水疗馆", "000900090004": "幼儿园", "00090001000100300008": "其他市场",
     "000900140005": "水塘", "00090022": "祠堂", "000900120002": "工厂车间", "000900130007": "游戏室", "000900130017": "夜总会",
     "000900070001": "公交车站", "000900130018": "KTV", "000900130023": "休闲中心", "000900070007": "码头", "000900130011": "网吧",
     "0009000100010029": "商场", "000900010011": "写字楼", "000900030004": "人行天桥", "000900120010": "其他",
     "0009000100040003": "其他", "0009000100020009": "酒吧", "000900130013": "发廊", "000900140011": "其他",
     "000900120007": "停车场", "000900130002": "桑拿", "0009000100010005": "药店", "000900010009": "加油场所",
     "000900130022": "彩票店", "000900140001": "菜地", "0009000100050012": "其他", "0009000100030004": "日租公寓",
     "000900140002": "树林", "000900030001": "机动车道（交通灯）", "000900050003": "集体宿舍", "00090017": "党政军机关",
     "0009000100040002": "车库", "000900140010": "榕树下", "0009000100040001": "露天", "000900090003": "中小学",
     "000900100002": "门诊、急诊部(含诊室内及候诊区)", "0009000100010016": "五金店", "000900070003": "汽车客运站", "0009001500010004": "QQ",
     "000900070005": "地铁口", "000900090005": "其他", "000900150009": "其他", "000900110003": "文化广场", "00090019": "车管所",
     "000900130008": "健身房", "000900020005": "其他", "0009000100010002": "服装店", "000900060001": "物业小区",
     "000900130016": "养生馆/SPA", "0009000100010020": "糖烟酒店", "000900050007": "非围蔽小区", "0009000100020004": "快餐店",
     "000900060009": "仓库厂房", "000900140004": "果园", "000900070009": "其他", "000900120001": "办公楼",
     "0009000100020014": "饭庄", "0009000100020007": "大排档", "0009000100060001": "足球场", "000900010008": "废品收购场所",
     "0009000100020008": "咖啡屋", "000900050009": "停车场", "000900120006": "值班室", "0009000100010019": "茶叶店",
     "0009000100050003": "电信营业厅", "000900140006": "水沟", "0009000100010010": "日用品", "0009000100010021": "家具店",
     "0009001500010001": "微信", "000900030006": "绿化带", "0009000100030003": "旅店", "000900070004": "地铁站",
     "00090021": "事业单位", "0009000100060008": "其他", "000900130005": "电影院", "0009000100010008": "手机店",
     "0009000100070001": "汽车修理厂", "000900070006": "BRT站", "000900110006": "宗教场所", "0009000300080005": "其他",
     "000900120003": "仓库", "000900120004": "宿舍", "0009000100070004": "摩托车修理店", "0009000100020006": "酒家",
     "0009000100030007": "其他", "0009000100020011": "面包店", "0009000100070002": "汽车美容/洗车店", "000900050005": "机关大院",
     "000900140008": "简易工棚", "0009000900010003": "学生宿舍楼", "0009000100050002": "邮局", "0009000100010017": "报刊亭",
     "0009000900010004": "其他", "0009000100020001": "麦当劳", "000900030005": "人行隧道", "0009000100030005": "地下旅店",
     "000900120009": "矿场", "0009000100020010": "茶艺馆", "000900010012": "培训机构", "000900100001": "病房",
     "000900240008": "其他市场", "000900060003": "集体宿舍", "0009000100050001": "房产中介", "000900150007": "彩票网站",
     "0009000100070005": "家电维修店", "000900080016": "船", "000900070008": "机场", "000900120008": "路面",
     "0009000100020002": "肯德基", "000900130014": "美甲店", "000900240007": "批发市场", "000900030003": "自行车道",
     "000900240004": "物流市场", "000900250001": "在建楼盘", "00090001000100300005": "快递公司", "0009000100050011": "会展中心",
     "00090001000100300001": "布匹市场", "000900250002": "装修工程", "0009000100060003": "羽毛球场", "000900240003": "茶叶市场",
     "000900240005": "快递公司", "000900060010": "写字楼", "000900140003": "路面", "00090001000100300003": "茶叶市场",
     "0009000100010004": "饰物店", "0009000100070008": "其他维修店", "000900140007": "水田", "0009000100010007": "电器",
     "0009000100070003": "自行车修理店", "000900020002": "ATM机", "000900080003": "私家车", "000900090002": "中专技校",
     "000900250003": "绿化工程", "0009000300080001": "服务区", "0009000300080004": "机动车道", "0009000100010003": "鞋店",
     "000900100011": "家属楼", "00090001000100300004": "物流市场", "00090001000100300007": "批发市场", "000900110004": "观光休闲带",
     "0009000100010013": "家具专柜", "000900080001": "公交车", "000900060005": "机关大院", "000900150002": "网络游戏",
     "0009000100030001": "招待所", "000900110002": "收费公园（含游乐场、主题公园）", "000900080005": "出租车", "0009001500040006": "淘宝网",
     "000900130019": "卡拉OK", "0009000300080002": "隧道涵洞", "000900130020": "练歌房", "0009001500040007": "其他"}
# predictor3 = None
# predictor4 = None
# predictor5 = None
def loadmodel():
    model1 = torch.load("17.model")
    # model2 = torch.load("标签备用.model")
    # model3 = torch.load("作案手段change.seq2seq.model")
    # model4 = torch.load("案发场所change.seq2seq.model")
    # model5 = torch.load("警情类别.seq2seq.model")
    input_vocab1 = torch.load("input_17.vocab")
    # input_vocab2 = torch.load("标签备用.input_vocab")
    # input_vocab3 = torch.load("作案手段change.input_vocab")
    # input_vocab4 = torch.load("案发场所change.input_vocab")
    # input_vocab5 = torch.load("警情类别.input_vocab")
    output_vocab1 = torch.load("output_17.vocab")
    # output_vocab2 = torch.load("标签备用.output_vocab")
    # output_vocab3 = torch.load("作案手段change.output_vocab")
    # output_vocab4 = torch.load("案发场所change.output_vocab")
    # output_vocab5 = torch.load("警情类别.output_vocab")
    global predictor1
    # global predictor2
    # global predictor3
    # global predictor4
    # global predictor5
    predictor1 = Predictor(model1, input_vocab1, output_vocab1)
    # predictor2 = Predictor(model2, input_vocab2, output_vocab2)
    # predictor3 = Predictor(model3, input_vocab3, output_vocab3)
    # predictor4 = Predictor(model4, input_vocab4, output_vocab4)
    # predictor5 = Predictor(model5, input_vocab5, output_vocab5)

def get_predict_result(text):
    cut_word = jieba.cut(str(text), cut_all=True, HMM=False)
    seg = ' '.join(cut_word)
    seq1 = seg.strip().split()
    print(seq1)
    seq = seq1[:-1]

    result = {"发案场所": [{"content": predictor1.predict(seq)['tgt_seq'][0], "score": predictor1.predict(seq)['score'][0],
              "datail": det[predictor1.predict(seq)['tgt_seq'][0]]}]}
    # result = {"预测结果": {"标签": [{"content": predictor2.predict(seq)['tgt_seq'][:-1], "score": predictor2.predict(seq)['score'][0],"location": "undefined"}],
    #                    "损失财物": [{"content": predictor1.predict(seq)['tgt_seq'][:-1], "score": predictor1.predict(seq)['score'][0],"location": "undefined"}],
    #                    "作案手段": [{"content": predictor3.predict(seq)[0].replace(' ', '-'), "location": "undefined"}],
    #                    "案发场所": [{"content": predictor4.predict(seq)[0].replace(' ', '-'), "location": "undefined"}],
    #                    "警情类别": [{"content": predictor5.predict(seq)[0].replace(' ', '-'), "location": "undefined"}],
    #                }}

    # result = {"预测结果": {"标签": [{"content": predictor2.predict(seq), "location": "undefined"}],
    #                    "损失财物": [{"content": predictor1.predict(seq), "location": "undefined"}],
    #                    "作案手段": [{"content": predictor3.predict(seq), "location": "undefined"}],
    #                    "案发场所": [{"content": predictor4.predict(seq), "location": "undefined"}],
    #                    "警情类别": [{"content": predictor5.predict(seq), "location": "undefined"}],
    #                    }}
    # result=predictor3.predict(seq)
    return result

if __name__ == "__main__":
    import os

    print(os.getcwd())
    loadmodel()
    import time
    print(time.time())
    with open("F:\\work\\20.txt", 'w', encoding='utf-8') as f:
        with open("F:\\work\\17.txt", 'r', encoding='utf-8') as f0:
            for n in f0.readlines():
                a = get_predict_result(n)
                # a = get_predict_result('于2017年8月31日晚上，我所民警工作发现在广州市花都区狮岭镇合成村新成一队一房屋内有多人以三公大吃小的方式参与赌博，并当场抓获涉嫌开设赌场的胡有强。经查，胡有强有重大作案嫌疑。')
                    # b = input()
                # while True:
                #     print(time.time())
                #     a = get_predict_result("2018年3月4日9时26分， 报警人（管文山，男，湖南省人，身份证号码是：430721196512047353）报称：在广东省广州市白云区江高镇联滘路一横路弘力工业园1号仓库被撬门盗窃一批鞋子，自称价值约40000元人民币；一台蓝色女装电动车，车架号和电机号不清楚，于2017年10月份以1000元人民币购买，发票在车上一起被盗了；一台电焊机，于2017年12月份以1000元人民币购买，没有发票。（何梓轩）")
                #     # b=input('qinshuru:')
                #
                #     # a = get_predict_result(b)
                #     print(a)
                #     print(time.time())
                f.write(n)
                # f.write('\n')
                f.write(str(a))
                f.write('\n')
                print(a)
    print(time.time())
