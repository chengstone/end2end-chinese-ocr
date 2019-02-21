#coding:utf-8

import numpy as np
import random

import os
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL import ImageFont
from PIL.ImageFont import truetype

from skimage.util import random_noise
from skimage import img_as_float
from math import *
import cv2
import datetime, time
from tensorflow.python.ops import summary_ops_v2


import tensorflow as tf

# pip install tf-nightly-gpu-2.0-preview
# require compute capabilities >= 3.5
#         cuda 10

class ocr_network(object):
    def __init__(self, birnn_type=2, lstm_num_layers=1, batch_size=32, max_timestep=None):
        self.save_dir = "./models"

        self.table = []
        for i in range(256):
            self.table.append(i * 1.97)

        # Variables
        self.MIN_LEN = 4
        self.MAX_LEN = 6
        self.char_nums = 0
        self.filters_size = [32, 64, 128, 128]    # or [32, 64, 128, 256], [64, 128, 256, 512]
        self.cnn_layer_num = 4
        self.batch_size = batch_size
        self.channels = 1
        self.img_height = 64
        self.img_width = self.img_height

        if max_timestep is None:
            pass
            # self.max_timestep = tf.placeholder(tf.int32, name="max_timestep")
        else:
            self.max_timestep = max_timestep
        self.num_hidden = 128  # or 256 , 512

        self.lstm_num_layers = lstm_num_layers
        # 1:static_bidirectional_rnn  2:bidirectional_dynamic_rnn
        # 3:stack_bidirectional_dynamic_rnn  4:stack_bidirectional_rnn
        # 0:dynamic_rnn
        self.birnn_type = birnn_type
        # self.charset="1234567890"
        # self.charset = "啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂"
        # 一级汉字
        self.charset = "啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座"
        # 一级汉字 + 二级汉字
        # self.charset = "啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座亍丌兀丐廿卅丕亘丞鬲孬噩丨禺丿匕乇夭爻卮氐囟胤馗毓睾鼗丶亟鼐乜乩亓芈孛啬嘏仄厍厝厣厥厮靥赝匚叵匦匮匾赜卦卣刂刈刎刭刳刿剀剌剞剡剜蒯剽劂劁劐劓冂罔亻仃仉仂仨仡仫仞伛仳伢佤仵伥伧伉伫佞佧攸佚佝佟佗伲伽佶佴侑侉侃侏佾佻侪佼侬侔俦俨俪俅俚俣俜俑俟俸倩偌俳倬倏倮倭俾倜倌倥倨偾偃偕偈偎偬偻傥傧傩傺僖儆僭僬僦僮儇儋仝氽佘佥俎龠汆籴兮巽黉馘冁夔勹匍訇匐凫夙兕亠兖亳衮袤亵脔裒禀嬴蠃羸冫冱冽冼凇冖冢冥讠讦讧讪讴讵讷诂诃诋诏诎诒诓诔诖诘诙诜诟诠诤诨诩诮诰诳诶诹诼诿谀谂谄谇谌谏谑谒谔谕谖谙谛谘谝谟谠谡谥谧谪谫谮谯谲谳谵谶卩卺阝阢阡阱阪阽阼陂陉陔陟陧陬陲陴隈隍隗隰邗邛邝邙邬邡邴邳邶邺邸邰郏郅邾郐郄郇郓郦郢郜郗郛郫郯郾鄄鄢鄞鄣鄱鄯鄹酃酆刍奂劢劬劭劾哿勐勖勰叟燮矍廴凵凼鬯厶弁畚巯坌垩垡塾墼壅壑圩圬圪圳圹圮圯坜圻坂坩垅坫垆坼坻坨坭坶坳垭垤垌垲埏垧垴垓垠埕埘埚埙埒垸埴埯埸埤埝堋堍埽埭堀堞堙塄堠塥塬墁墉墚墀馨鼙懿艹艽艿芏芊芨芄芎芑芗芙芫芸芾芰苈苊苣芘芷芮苋苌苁芩芴芡芪芟苄苎芤苡茉苷苤茏茇苜苴苒苘茌苻苓茑茚茆茔茕苠苕茜荑荛荜茈莒茼茴茱莛荞茯荏荇荃荟荀茗荠茭茺茳荦荥荨茛荩荬荪荭荮莰荸莳莴莠莪莓莜莅荼莶莩荽莸荻莘莞莨莺莼菁萁菥菘堇萘萋菝菽菖萜萸萑萆菔菟萏萃菸菹菪菅菀萦菰菡葜葑葚葙葳蒇蒈葺蒉葸萼葆葩葶蒌蒎萱葭蓁蓍蓐蓦蒽蓓蓊蒿蒺蓠蒡蒹蒴蒗蓥蓣蔌甍蔸蓰蔹蔟蔺蕖蔻蓿蓼蕙蕈蕨蕤蕞蕺瞢蕃蕲蕻薤薨薇薏蕹薮薜薅薹薷薰藓藁藜藿蘧蘅蘩蘖蘼廾弈夼奁耷奕奚奘匏尢尥尬尴扌扪抟抻拊拚拗拮挢拶挹捋捃掭揶捱捺掎掴捭掬掊捩掮掼揲揸揠揿揄揞揎摒揆掾摅摁搋搛搠搌搦搡摞撄摭撖摺撷撸撙撺擀擐擗擤擢攉攥攮弋忒甙弑卟叱叽叩叨叻吒吖吆呋呒呓呔呖呃吡呗呙吣吲咂咔呷呱呤咚咛咄呶呦咝哐咭哂咴哒咧咦哓哔呲咣哕咻咿哌哙哚哜咩咪咤哝哏哞唛哧唠哽唔哳唢唣唏唑唧唪啧喏喵啉啭啁啕唿啐唼唷啖啵啶啷唳唰啜喋嗒喃喱喹喈喁喟啾嗖喑啻嗟喽喾喔喙嗪嗷嗉嘟嗑嗫嗬嗔嗦嗝嗄嗯嗥嗲嗳嗌嗍嗨嗵嗤辔嘞嘈嘌嘁嘤嘣嗾嘀嘧嘭噘嘹噗嘬噍噢噙噜噌噔嚆噤噱噫噻噼嚅嚓嚯囔囗囝囡囵囫囹囿圄圊圉圜帏帙帔帑帱帻帼帷幄幔幛幞幡岌屺岍岐岖岈岘岙岑岚岜岵岢岽岬岫岱岣峁岷峄峒峤峋峥崂崃崧崦崮崤崞崆崛嵘崾崴崽嵬嵛嵯嵝嵫嵋嵊嵩嵴嶂嶙嶝豳嶷巅彳彷徂徇徉後徕徙徜徨徭徵徼衢彡犭犰犴犷犸狃狁狎狍狒狨狯狩狲狴狷猁狳猃狺狻猗猓猡猊猞猝猕猢猹猥猬猸猱獐獍獗獠獬獯獾舛夥飧夤夂饣饧饨饩饪饫饬饴饷饽馀馄馇馊馍馐馑馓馔馕庀庑庋庖庥庠庹庵庾庳赓廒廑廛廨廪膺忄忉忖忏怃忮怄忡忤忾怅怆忪忭忸怙怵怦怛怏怍怩怫怊怿怡恸恹恻恺恂恪恽悖悚悭悝悃悒悌悛惬悻悱惝惘惆惚悴愠愦愕愣惴愀愎愫慊慵憬憔憧憷懔懵忝隳闩闫闱闳闵闶闼闾阃阄阆阈阊阋阌阍阏阒阕阖阗阙阚丬爿戕氵汔汜汊沣沅沐沔沌汨汩汴汶沆沩泐泔沭泷泸泱泗沲泠泖泺泫泮沱泓泯泾洹洧洌浃浈洇洄洙洎洫浍洮洵洚浏浒浔洳涑浯涞涠浞涓涔浜浠浼浣渚淇淅淞渎涿淠渑淦淝淙渖涫渌涮渫湮湎湫溲湟溆湓湔渲渥湄滟溱溘滠漭滢溥溧溽溻溷滗溴滏溏滂溟潢潆潇漤漕滹漯漶潋潴漪漉漩澉澍澌潸潲潼潺濑濉澧澹澶濂濡濮濞濠濯瀚瀣瀛瀹瀵灏灞宀宄宕宓宥宸甯骞搴寤寮褰寰蹇謇辶迓迕迥迮迤迩迦迳迨逅逄逋逦逑逍逖逡逵逶逭逯遄遑遒遐遨遘遢遛暹遴遽邂邈邃邋彐彗彖彘尻咫屐屙孱屣屦羼弪弩弭艴弼鬻屮妁妃妍妩妪妣妗姊妫妞妤姒妲妯姗妾娅娆姝娈姣姘姹娌娉娲娴娑娣娓婀婧婊婕娼婢婵胬媪媛婷婺媾嫫媲嫒嫔媸嫠嫣嫱嫖嫦嫘嫜嬉嬗嬖嬲嬷孀尕尜孚孥孳孑孓孢驵驷驸驺驿驽骀骁骅骈骊骐骒骓骖骘骛骜骝骟骠骢骣骥骧纟纡纣纥纨纩纭纰纾绀绁绂绉绋绌绐绔绗绛绠绡绨绫绮绯绱绲缍绶绺绻绾缁缂缃缇缈缋缌缏缑缒缗缙缜缛缟缡缢缣缤缥缦缧缪缫缬缭缯缰缱缲缳缵幺畿巛甾邕玎玑玮玢玟珏珂珑玷玳珀珉珈珥珙顼琊珩珧珞玺珲琏琪瑛琦琥琨琰琮琬琛琚瑁瑜瑗瑕瑙瑷瑭瑾璜璎璀璁璇璋璞璨璩璐璧瓒璺韪韫韬杌杓杞杈杩枥枇杪杳枘枧杵枨枞枭枋杷杼柰栉柘栊柩枰栌柙枵柚枳柝栀柃枸柢栎柁柽栲栳桠桡桎桢桄桤梃栝桕桦桁桧桀栾桊桉栩梵梏桴桷梓桫棂楮棼椟椠棹椤棰椋椁楗棣椐楱椹楠楂楝榄楫榀榘楸椴槌榇榈槎榉楦楣楹榛榧榻榫榭槔榱槁槊槟榕槠榍槿樯槭樗樘橥槲橄樾檠橐橛樵檎橹樽樨橘橼檑檐檩檗檫猷獒殁殂殇殄殒殓殍殚殛殡殪轫轭轱轲轳轵轶轸轷轹轺轼轾辁辂辄辇辋辍辎辏辘辚軎戋戗戛戟戢戡戥戤戬臧瓯瓴瓿甏甑甓攴旮旯旰昊昙杲昃昕昀炅曷昝昴昱昶昵耆晟晔晁晏晖晡晗晷暄暌暧暝暾曛曜曦曩贲贳贶贻贽赀赅赆赈赉赇赍赕赙觇觊觋觌觎觏觐觑牮犟牝牦牯牾牿犄犋犍犏犒挈挲掰搿擘耄毪毳毽毵毹氅氇氆氍氕氘氙氚氡氩氤氪氲攵敕敫牍牒牖爰虢刖肟肜肓肼朊肽肱肫肭肴肷胧胨胩胪胛胂胄胙胍胗朐胝胫胱胴胭脍脎胲胼朕脒豚脶脞脬脘脲腈腌腓腴腙腚腱腠腩腼腽腭腧塍媵膈膂膑滕膣膪臌朦臊膻臁膦欤欷欹歃歆歙飑飒飓飕飙飚殳彀毂觳斐齑斓於旆旄旃旌旎旒旖炀炜炖炝炻烀炷炫炱烨烊焐焓焖焯焱煳煜煨煅煲煊煸煺熘熳熵熨熠燠燔燧燹爝爨灬焘煦熹戾戽扃扈扉礻祀祆祉祛祜祓祚祢祗祠祯祧祺禅禊禚禧禳忑忐怼恝恚恧恁恙恣悫愆愍慝憩憝懋懑戆肀聿沓泶淼矶矸砀砉砗砘砑斫砭砜砝砹砺砻砟砼砥砬砣砩硎硭硖硗砦硐硇硌硪碛碓碚碇碜碡碣碲碹碥磔磙磉磬磲礅磴礓礤礞礴龛黹黻黼盱眄眍盹眇眈眚眢眙眭眦眵眸睐睑睇睃睚睨睢睥睿瞍睽瞀瞌瞑瞟瞠瞰瞵瞽町畀畎畋畈畛畲畹疃罘罡罟詈罨罴罱罹羁罾盍盥蠲钅钆钇钋钊钌钍钏钐钔钗钕钚钛钜钣钤钫钪钭钬钯钰钲钴钶钷钸钹钺钼钽钿铄铈铉铊铋铌铍铎铐铑铒铕铖铗铙铘铛铞铟铠铢铤铥铧铨铪铩铫铮铯铳铴铵铷铹铼铽铿锃锂锆锇锉锊锍锎锏锒锓锔锕锖锘锛锝锞锟锢锪锫锩锬锱锲锴锶锷锸锼锾锿镂锵镄镅镆镉镌镎镏镒镓镔镖镗镘镙镛镞镟镝镡镢镤镥镦镧镨镩镪镫镬镯镱镲镳锺矧矬雉秕秭秣秫稆嵇稃稂稞稔稹稷穑黏馥穰皈皎皓皙皤瓞瓠甬鸠鸢鸨鸩鸪鸫鸬鸲鸱鸶鸸鸷鸹鸺鸾鹁鹂鹄鹆鹇鹈鹉鹋鹌鹎鹑鹕鹗鹚鹛鹜鹞鹣鹦鹧鹨鹩鹪鹫鹬鹱鹭鹳疒疔疖疠疝疬疣疳疴疸痄疱疰痃痂痖痍痣痨痦痤痫痧瘃痱痼痿瘐瘀瘅瘌瘗瘊瘥瘘瘕瘙瘛瘼瘢瘠癀瘭瘰瘿瘵癃瘾瘳癍癞癔癜癖癫癯翊竦穸穹窀窆窈窕窦窠窬窨窭窳衤衩衲衽衿袂袢裆袷袼裉裢裎裣裥裱褚裼裨裾裰褡褙褓褛褊褴褫褶襁襦襻疋胥皲皴矜耒耔耖耜耠耢耥耦耧耩耨耱耋耵聃聆聍聒聩聱覃顸颀颃颉颌颍颏颔颚颛颞颟颡颢颥颦虍虔虬虮虿虺虼虻蚨蚍蚋蚬蚝蚧蚣蚪蚓蚩蚶蛄蚵蛎蚰蚺蚱蚯蛉蛏蚴蛩蛱蛲蛭蛳蛐蜓蛞蛴蛟蛘蛑蜃蜇蛸蜈蜊蜍蜉蜣蜻蜞蜥蜮蜚蜾蝈蜴蜱蜩蜷蜿螂蜢蝽蝾蝻蝠蝰蝌蝮螋蝓蝣蝼蝤蝙蝥螓螯螨蟒蟆螈螅螭螗螃螫蟥螬螵螳蟋蟓螽蟑蟀蟊蟛蟪蟠蟮蠖蠓蟾蠊蠛蠡蠹蠼缶罂罄罅舐竺竽笈笃笄笕笊笫笏筇笸笪笙笮笱笠笥笤笳笾笞筘筚筅筵筌筝筠筮筻筢筲筱箐箦箧箸箬箝箨箅箪箜箢箫箴篑篁篌篝篚篥篦篪簌篾篼簏簖簋簟簪簦簸籁籀臾舁舂舄臬衄舡舢舣舭舯舨舫舸舻舳舴舾艄艉艋艏艚艟艨衾袅袈裘裟襞羝羟羧羯羰羲籼敉粑粝粜粞粢粲粼粽糁糇糌糍糈糅糗糨艮暨羿翎翕翥翡翦翩翮翳糸絷綦綮繇纛麸麴赳趄趔趑趱赧赭豇豉酊酐酎酏酤酢酡酰酩酯酽酾酲酴酹醌醅醐醍醑醢醣醪醭醮醯醵醴醺豕鹾趸跫踅蹙蹩趵趿趼趺跄跖跗跚跞跎跏跛跆跬跷跸跣跹跻跤踉跽踔踝踟踬踮踣踯踺蹀踹踵踽踱蹉蹁蹂蹑蹒蹊蹰蹶蹼蹯蹴躅躏躔躐躜躞豸貂貊貅貘貔斛觖觞觚觜觥觫觯訾謦靓雩雳雯霆霁霈霏霎霪霭霰霾龀龃龅龆龇龈龉龊龌黾鼋鼍隹隼隽雎雒瞿雠銎銮鋈錾鍪鏊鎏鐾鑫鱿鲂鲅鲆鲇鲈稣鲋鲎鲐鲑鲒鲔鲕鲚鲛鲞鲟鲠鲡鲢鲣鲥鲦鲧鲨鲩鲫鲭鲮鲰鲱鲲鲳鲴鲵鲶鲷鲺鲻鲼鲽鳄鳅鳆鳇鳊鳋鳌鳍鳎鳏鳐鳓鳔鳕鳗鳘鳙鳜鳝鳟鳢靼鞅鞑鞒鞔鞯鞫鞣鞲鞴骱骰骷鹘骶骺骼髁髀髅髂髋髌髑魅魃魇魉魈魍魑飨餍餮饕饔髟髡髦髯髫髻髭髹鬈鬏鬓鬟鬣麽麾縻麂麇麈麋麒鏖麝麟黛黜黝黠黟黢黩黧黥黪黯鼢鼬鼯鼹鼷鼽鼾齄"
        self.num_classes=len(self.charset)+2

        self.encode_maps = {}
        self.decode_maps = {}
        for i, char in enumerate(self.charset, 1):
            self.encode_maps[char] = i
            self.decode_maps[i] = char

        SPACE_INDEX = 0
        SPACE_TOKEN = ''
        self.encode_maps[SPACE_TOKEN] = SPACE_INDEX
        self.decode_maps[SPACE_INDEX] = SPACE_TOKEN

        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        self.distributed_train = lambda it: self.strategy.experimental_run(self.train_step, it)
        self.distributed_train = tf.function(self.distributed_train)

        with self.strategy.scope():
            self.inputs_ = tf.keras.layers.Input(shape=[self.img_height, None, self.channels], dtype="float32", name='inputs')

            with tf.name_scope('cnn'):
                self.layer = self.inputs_
                for i in range(self.cnn_layer_num):
                    with tf.name_scope('cnn_layer-%d' % i):
                        self.layer = self.cnn_layer(self.layer, self.filters_size[i])
                        print(self.layer.get_shape())

            _, feature_h, feature_w, cnn_out_channels = self.layer.get_shape().as_list()
            with tf.name_scope('lstm'):

                if self.birnn_type == 0:  # dynamic_rnn
                    # [batch_size, feature_w, feature_h, cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                    # `feature_w` is max_timestep in lstm.  # feature_w(self.max_timestep) unknown
                    self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                    print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                    # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
                    self.cell1 = tf.keras.layers.LSTMCell(self.num_hidden)
                    self.cell2 = tf.keras.layers.LSTMCell(self.num_hidden)

                    # [batch_size, max_timestep, self.num_hidden]
                    outputs = tf.keras.layers.RNN([self.cell1, self.cell2], return_sequences=True)(self.layer)

                    self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                    # Time major  [max_timestep, batch_size, num_classes]
                    self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                    self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                    self.model.summary()
                elif self.birnn_type == 1 and max_timestep is not None:  # static_bidirectional_rnn
                    # [batch_size, feature_w, feature_h, cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                    # `feature_w` is max_timestep in lstm.
                    # [batch_size, max_timestep, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Reshape([self.max_timestep, feature_h * cnn_out_channels])(self.layer)

                    print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                    self.cells = [tf.keras.layers.LSTMCell(self.num_hidden) for _ in range(self.lstm_num_layers)]
                    self.cells_stack = tf.keras.layers.StackedRNNCells(self.cells)
                    self.rnn_cells_stack = tf.keras.layers.RNN(self.cells_stack, return_sequences=True)

                    # layer: [max_timestep, batch_size, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.layer)

                    # outputs: [max_timestep, batch_size, 2*num_hidden]
                    outputs = tf.keras.layers.Bidirectional(self.rnn_cells_stack)(self.layer)

                    # Time major  [max_timestep, batch_size, num_classes]
                    self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                    self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                    self.model.summary()
                elif self.birnn_type == 2:  # bidirectional_dynamic_rnn
                    # [batch_size, feature_w, feature_h, cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                    # `feature_w` is max_timestep in lstm.
                    # [batch_size, max_timestep, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                    print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                    self.cells = [tf.keras.layers.LSTMCell(self.num_hidden) for _ in range(self.lstm_num_layers)]
                    self.cells_stack = tf.keras.layers.StackedRNNCells(self.cells)
                    self.rnn_cells_stack = tf.keras.layers.RNN(self.cells_stack, return_sequences=True)

                    # [batch_size, max_timestep, 2 * num_hidden]
                    outputs = tf.keras.layers.Bidirectional(self.rnn_cells_stack)(self.layer)

                    # [batch_size, max_timestep, num_classes]
                    self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                    # Time major  [max_timestep, batch_size, num_classes]
                    self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                    self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                    self.model.summary()
                elif self.birnn_type == 3:  # stack_bidirectional_dynamic_rnn
                    # [batch_size, feature_w, feature_h, cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                    # `feature_w` is max_timestep in lstm.
                    # [batch_size, max_timestep, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                    print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                    # [batch_size, max_timestep, 2 * num_hidden]
                    for _ in range(self.lstm_num_layers):
                        self.lstm_cell = tf.keras.layers.LSTM(self.num_hidden, return_sequences=True)
                        self.layer = tf.keras.layers.Bidirectional(self.lstm_cell)(self.layer)

                    # [batch_size, max_timestep, num_classes]
                    self.logits = tf.keras.layers.Dense(self.num_classes)(self.layer)

                    # Time major  [max_timestep, batch_size, num_classes]
                    self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                    self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                    self.model.summary()
                elif self.birnn_type == 4 and max_timestep is not None:  # stack_bidirectional_rnn
                    # [batch_size, feature_w, feature_h, cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                    # `feature_w` is max_timestep in lstm.
                    # [batch_size, max_timestep, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Reshape([self.max_timestep, feature_h * cnn_out_channels])(self.layer)

                    print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                    # layer: [max_timestep, batch_size, feature_h * cnn_out_channels]
                    self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.layer)

                    # outputs: [max_timestep, batch_size, 2*num_hidden]
                    for _ in range(self.lstm_num_layers):
                        self.lstm_cell = tf.keras.layers.LSTM(self.num_hidden, return_sequences=True)
                        self.layer = tf.keras.layers.Bidirectional(self.lstm_cell)(self.layer)

                    # Time major  [max_timestep, batch_size, num_classes]
                    self.logits = tf.keras.layers.Dense(self.num_classes)(self.layer)

                    self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                    self.model.summary()


            self.optimizer = tf.keras.optimizers.Adam(1e-3)  # decay=0.98

            if tf.io.gfile.exists(self.save_dir):
                #             print('Removing existing model dir: {}'.format(self.save_dir))
                #             tf.io.gfile.rmtree(self.save_dir)
                pass
            else:
                tf.io.gfile.makedirs(self.save_dir)

            train_dir = os.path.join(self.save_dir, 'summaries', 'train')
            test_dir = os.path.join(self.save_dir, 'summaries', 'eval')

            #         self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
            #         self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

            checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
            self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
            self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

            # Restore variables on creation if a checkpoint exists.
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

            self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            self.avg_acc = tf.keras.metrics.Mean('acc', dtype=tf.float32)
            self.avg_label_error_rate = tf.keras.metrics.Mean('ler', dtype=tf.float32)
        # self.learning_rate = tf.train.exponential_decay(1e-3,
        #                                                self.global_step,
        #                                                     10000,
        #                                                     0.98,
        #                                                staircase=True)
        # tf.summary.scalar('learning_rate', self.learning_rate)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
        #                                            momentum=0.9).minimize(self.loss,  # self.momentum
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
        #                                             momentum=0.9,  # self.momentum
        #                                             use_nesterov=True).minimize(self.loss,
        #                                                                         global_step=self.global_step)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                         beta1=0.9,  # self.beta1
        #                                         beta2=0.999).minimize(self.loss,  # self.beta2
        #                                                             global_step=self.global_step)

    def cnn_layer(self, layer, filter_size):
        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=filter_size, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(layer)
        layer = tf.keras.layers.ELU()(layer)
        layer = tf.keras.layers.MaxPool2D(strides=2, padding="same")(layer)

        return layer

    # def train_restore(self):
    #     if not os.path.isdir(self.save_dir):
    #         os.mkdir(self.save_dir)
    #     checkpoint = tf.train.get_checkpoint_state(self.save_dir)
    #
    #     if checkpoint and checkpoint.model_checkpoint_path:
    #         # self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
    #         self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
    #         print("Successfully loaded:", tf.train.latest_checkpoint(self.save_dir))
    #         # print("Successfully loaded:", checkpoint.model_checkpoint_path)
    #     else:
    #         print("Could not find old network weights")

    # def restore(self, file):
    #     print("Restoring from {0}".format(file))
    #     self.saver.restore(self.sess, file)  # self.ckpt

    def save(self, in_global_step=None):
        with self.strategy.scope():
            self.checkpoint.save(self.checkpoint_prefix)

        print("Model saved in file: {}".format(self.checkpoint_prefix))

    def gen_rand(self):
        buf = ""

        for i in range(self.char_nums):
            buf += random.choice(self.charset)
        return buf

    def random_color(self, start, end, opacity=None):
        red = random.randint(start, end)
        green = random.randint(start, end)
        blue = random.randint(start, end)
        if opacity is None:
            return (red, green, blue)
        return (red, green, blue, opacity)

    def _draw_character(self, c, font_path, draw, color):
        # font = random.choice(self.truefonts)
        # font = ImageFont.truetype(font_path, int(self.img_height * random.uniform(0.6, 1.1)), )
        font = ImageFont.truetype(font_path, self.img_height, )
        w, h = draw.textsize(c, font=font)

        dx = random.randint(0, 4)
        dy = random.randint(0, 6)
        im = Image.new('RGBA', (w + dx, h + dy))
        Draw(im).text((dx, dy), c, font=font, fill=color)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

        # warp
        dx = w * random.uniform(0.1, 0.3)
        dy = h * random.uniform(0.2, 0.3)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), Image.QUAD, data)
        return im

    def generate_image(self, chars, font_path):
        background = self.random_color(0, 0)
        color = self.random_color(255, 255)  # , random.randint(220, 255)

        image = Image.new("RGB", (self.img_width, self.img_height), background)
        draw = Draw(image)

        # draw.text((x, y), char, (0, 0, 0),
        #           font=font)

        images = []
        for c in chars:
            images.append(self._draw_character(c, font_path, draw, color))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self.img_width)
        image = image.resize((width, self.img_height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(self.table)
            image.paste(im, (offset, int((self.img_height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        # if width > self.img_width:
        #     image = image.resize((self.img_width, self.img_height))

        return image

    def imgaug_process(self, data):
        # do something here
        return np.array(data, dtype=np.uint8)

    def generateImg(self):
        font_list = ["fonts/fangzheng_fangsong.ttf"]  # add more fonts here
        font_nums = len(font_list)
        font_index = random.randint(0, font_nums-1)
        font_path = font_list[font_index]

        if not os.path.exists(font_path):
            print('cannot open the font')
        theChars=self.gen_rand()
        data = self.generate_image(theChars, font_path)
        return np.array(data),theChars

    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return tf.sparse.SparseTensor(indices, values, shape)  # (indices, values, shape)

    def get_batches(self, imgaug_process=False):
        images = []
        labels = []
        self.char_nums = random.randint(self.MIN_LEN, self.MAX_LEN)

        while True:
            im, label = self.generateImg()
            # print(im.shape)
            if imgaug_process:
                im = self.imgaug_process(im)

            if len(im.shape) > 2:
                im = Image.fromarray(im).convert('L')
                im = np.array(im, dtype=np.uint8)

            if im.shape[0] > self.img_height:
                im = im[int((im.shape[0] - self.img_height) / 2):int(
                    (im.shape[0] - self.img_height) / 2) + self.img_height, :]
            elif im.shape[0] < self.img_height:
                dst = np.reshape(np.array([0] * (im.shape[1] * int(self.img_height)), dtype=np.uint8),
                                 [int(self.img_height), im.shape[1]]).astype(np.uint8)
                dst[int((self.img_height - im.shape[0]) / 2):int((self.img_height - im.shape[0]) / 2) + im.shape[0],
                :] = im
                im = dst

            if im.shape[1] > self.char_nums * self.img_width:
                im = im[:, int((im.shape[1] - self.char_nums * self.img_width) / 2):int(
                    (im.shape[1] - self.char_nums * self.img_width) / 2) + self.char_nums * self.img_width]
            elif im.shape[1] < self.char_nums * self.img_width:
                dst = np.reshape(
                    np.array([0] * (self.char_nums * self.img_width * int(self.img_height)), dtype=np.uint8),
                    [int(self.img_height), self.char_nums * self.img_width]).astype(np.uint8)

                dst[:, int((self.char_nums * self.img_width - im.shape[1]) / 2):int(
                    (self.char_nums * self.img_width - im.shape[1]) / 2) + im.shape[1]] = im
                im = dst

            im=np.expand_dims(im, 2)
            # print(im.shape)
            # cv2.imwrite("./data/train/{}.png".format(label), im)

            images.append(im.astype(np.float32))

            code = [self.encode_maps[c] for c in list(label)]

            labels.append(code)

            if len(images) == self.batch_size:
                batch_seq_len = np.asarray([(self.char_nums * self.img_width) // 16] * self.batch_size, dtype=np.int32)
                batch_inputs = np.array(images)
                batch_labels = self.sparse_tuple_from_label(labels)

                yield batch_inputs, batch_seq_len, np.array(labels), batch_labels

                self.char_nums = random.randint(self.MIN_LEN, self.MAX_LEN)

                images = []
                labels = []

    def accuracy_calculation(self, original_seq, decoded_seq, ignore_value=-1, isPrint=False):
        print(original_seq)
        print(decoded_seq)

        # if len(original_seq) != len(decoded_seq):
        #     print('original lengths({}) is different from the decoded_seq({}), please check again'.format(len(original_seq), len(decoded_seq)))
        #     return 0
        count = 0
        i = 0
        for origin_label in [tf.map_fn(lambda x: x, original_seq)]:
            decoded_label = [tf.map_fn(lambda y: y, decoded_seq)]
            print("decoded_label ", decoded_label)
            # decoded_label = [j.numpy() for j in decoded_seq[i] if j.numpy() != ignore_value]
            if isPrint:
                print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

                with open('./test.csv', 'w') as f:
                    f.write(str(origin_label) + '\t' + str(decoded_label))
                    f.write('\n')

            try:
                if len(origin_label) == len(decoded_label):
                    match_num = 0
                    for i, c in enumerate(origin_label):
                        if decoded_label[i] == c:
                            match_num += 1
                    if match_num == len(decoded_label):
                        count += 1
            except:
                pass
            i += 1
        return count * 1.0 / len(original_seq)

    def compute_loss(self, labels, logits, seq_len):

        with tf.name_scope("loss"):

            loss = tf.nn.ctc_loss(labels=labels,
                                  logits=logits,
                                  label_length=None, logit_length=seq_len, blank_index=-1)  # logits_time_major=False,

            loss = tf.reduce_mean(loss)


        return loss

    def compute_metrics(self, labels, logits, seq_len):

        # decoded, log_prob = \
        #     tf.nn.ctc_beam_search_decoder(inputs=logits,
        #                                   sequence_length=seq_len)

        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)  # this is faster

        # print(decoded[0], log_prob)
        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1)
        # print(dense_decoded)

        # ####Evaluating
        # self.logitsMaxTest = tf.slice(tf.argmax(self.logits, 2), [0, 0], [self.seq_len[0], 1])
        label_error_rate = tf.reduce_mean(
            input_tensor=tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

        return dense_decoded, label_error_rate

    # TODO(yashkatariya): Add tf.function when b/123315763 is resolved
    # @tf.function
    def train_step(self, it):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0

        batch_inputs = it[0]
        batch_labels = it[1]
        batch_seq_len = it[2]
        orig_labels = it[3]

        # print("batch_inputs ", batch_inputs)
        # print("batch_labels ", batch_labels)
        # print("batch_seq_len ", batch_seq_len)

        self.avg_loss.reset_states()
        self.avg_label_error_rate.reset_states()

        with tf.GradientTape() as tape:
            logits = self.model(batch_inputs, training=True)
        # import pickle
        # pickle.dump((batch_labels, logits, batch_seq_len), open('loss.p', 'wb'))
            loss = self.compute_loss(batch_labels, logits, batch_seq_len)
            # dense_decoded, label_error_rate = self.compute_metrics(batch_labels, logits, batch_seq_len)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        dense_decoded, label_error_rate = self.compute_metrics(batch_labels, logits, batch_seq_len)
        # accuracy = self.accuracy_calculation(orig_labels, dense_decoded,
        #                                      ignore_value=-1, isPrint=True)
        # print("dense_decoded ", dense_decoded)
        self.avg_loss(loss)
        # self.avg_acc(accuracy)
        self.avg_label_error_rate(label_error_rate)

        return self.avg_loss.result(), self.avg_label_error_rate.result()#, dense_decoded  #, self.avg_acc.result()

    def training(self, batch_inputs, batch_labels, batch_seq_len, orig_labels):

        start_time = time.time()
        #
        with self.strategy.scope():
            train_dataset = tf.data.Dataset.from_tensor_slices((batch_inputs, batch_labels, batch_seq_len, orig_labels)).batch(
                len(batch_seq_len))
            # .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
            train_iterator = self.strategy.make_dataset_iterator(train_dataset)
            train_iterator.initialize()
            loss, label_error_rate = self.distributed_train(train_iterator) # , accuracy

        # dense_decoded, label_error_rate = self.compute_metrics(batch_labels, self.logits, batch_seq_len)
        # accuracy = self.accuracy_calculation(orig_labels, dense_decoded,
        #                                      ignore_value=-1, isPrint=True)

        # summary_ops_v2.scalar('loss', loss)
        # summary_ops_v2.scalar('label_error_rate', label_error_rate)
        #
        # self.avg_loss(loss)
        # self.avg_acc(accuracy)
        # self.avg_label_error_rate(label_error_rate)


        now = datetime.datetime.now()
        log = "{}/{} {}:{}:{} global_step {}, " \
              "train_loss = {:.3f}, " \
              "label_error_rate = {:.3f}, train using time = {:.3f}"  # accuracy = {:.3f},

        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                         self.optimizer.iterations.numpy(), loss,
                         label_error_rate, time.time() - start_time))  # , accuracy
        if self.optimizer.iterations.numpy() % 10 == 0:
            # self.save(self.global_step)
            with self.strategy.scope():
                self.checkpoint.save(self.checkpoint_prefix)
            self.inference()

    def inference(self):
        # model = ocr_network(batch_size=1)
        im = cv2.imread("./newimg.png", cv2.IMREAD_GRAYSCALE)
        # im = ~im
        # ratio = 64 / im.shape[0]

        # im = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        # im[im > 100] = 255
        # cv2.imwrite("./reverse_scale_2.png", im)
        if im.shape[0] > self.img_height:
            im = im[int((im.shape[0] - self.img_height) / 2):int(
                (im.shape[0] - self.img_height) / 2) + self.img_height, :]
        elif im.shape[0] < self.img_height:
            dst = np.reshape(np.array([0] * (im.shape[1] * int(self.img_height)), dtype=np.uint8),
                             [int(self.img_height), im.shape[1]]).astype(np.uint8)
            dst[int((self.img_height - im.shape[0]) / 2):int((self.img_height - im.shape[0]) / 2) + im.shape[0],
            :] = im
            im = dst

        # print(im.shape)

        self.forward(im.astype(np.float32))

    def forward(self, img):
        im = np.expand_dims(img, 2)
        im = np.expand_dims(im, 0)

        logits = self.model(im, training=False)
        decoded, log_prob = \
            tf.nn.ctc_beam_search_decoder(inputs=logits,
                                          sequence_length=np.asarray([(img.shape[1]) // 16], dtype=np.int32))

        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, np.asarray([(img.shape[1]) // 16], dtype=np.int32), merge_repeated=False)
        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1)

        decoded_label = [self.decode_maps[j.numpy()] for j in dense_decoded[0] if j.numpy() != -1]
        print("-------- prediction : {} --------".format(decoded_label))


def train():
    model = ocr_network()

    train_batches = model.get_batches()
    try:
        while True:
            batch_inputs, batch_seq_len, orig_labels, batch_labels = next(train_batches)
            model.training(batch_inputs, batch_labels, batch_seq_len, orig_labels)
    except KeyboardInterrupt:
        model.save()


def inference():
    model = ocr_network(batch_size=1)
    im = cv2.imread("./newimg.png", cv2.IMREAD_GRAYSCALE)
    # im = ~im
    # ratio = 64 / im.shape[0]

    # im = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    # im[im > 100] = 255
    # cv2.imwrite("./reverse_scale_2.png", im)
    if im.shape[0] > model.img_height:
        im = im[int((im.shape[0] - model.img_height) / 2):int(
            (im.shape[0] - model.img_height) / 2) + model.img_height, :]
    elif im.shape[0] < model.img_height:
        dst = np.reshape(np.array([0] * (im.shape[1] * int(model.img_height)), dtype=np.uint8),
                         [int(model.img_height), im.shape[1]]).astype(np.uint8)
        dst[int((model.img_height - im.shape[0]) / 2):int((model.img_height - im.shape[0]) / 2) + im.shape[0],
        :] = im
        im = dst

    # print(im.shape)

    model.forward(im.astype(np.float32))

if __name__ == '__main__':
    train()
    # inference()
