# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False,kpt_num = 5,nc = 1):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
        self.kpt_num = kpt_num
        self.nc = nc
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        #定义分类损失(BCEcls)和置信度损失(BCEobj)和关键点置信度损失(BCE_kptv)
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        #标签平滑
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        #g=0代表不用focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            #g>0将分类损失和置信度损失换成focalloss损失函数
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        #det返回模型检测头 detector 对应产出的四个输出feature map
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        #balance设置 对应检测头 输出的置信度损失系数 (平衡三个feature map的置信度损失)
        #从左到右分别对应大尺寸输出(80*80，检测小目标)->小尺寸输出(20*20,检测大目标)
        #对小目标的检测加大特征图损失系数 让模型更注重小物体的检测
        #
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #三个预测头的下采样率 det.stride :[8,16,32]
        # .index(16):求出下采样率stride=16的索引
        #autobalance该参数会自动计算更新3个feature map的置信度损失系数self.balance
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        #self.BCEcls：类别损失 self.BCEobj置信度损失 self.hyp：超参数 self.gr：计算真实框的置信度标准的iou ratio. self.autobalance:是否自动更新各feature map的置信度损失平衡
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        """
        na： number of anchor 每个grid的anchor数量=3
        nc：num of classes 数据集类别 1
        nl: num of detect layer 检测头个数 4
        anchors:输出的feature map上的 anchor [3,3,3,3] 四个检测头上分别有三组(w,h)
        nkpt ： 关键点个数
        """
        for k in 'na', 'nc', 'nl', 'anchors', 'nkpt':
            #setattr:给对象的属性k赋值为 getattr(det,k)
            #getattr:返回det对象的k属性
            #将det的k属性赋值给self.k属性，其中k in 'na' 'nc 'nl 'anchor' 'nkpt'
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
         :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                    [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [4, 16] [num_object,  batch_index+class+xywh+kpt_num*2]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params torch.cat((lbox, lobj, lcls, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        device = targets.device
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89][:self.nkpt],device=device) / (self.nkpt/4)
        #tcls[4,6] 四个检测头对应的6个gtbox类别  indices[4,(6)ima_index,(6)anc_indxex,(6)gy,(6)gx]
        # anchor(4,[6,2])表示4个检测头对应的6个gt box所对应的anchor w、h。 tbox[4，[6,4]]
        """
        比如原始的gt box的中心坐标是(51.7, 44.8)，则该gt box由方格(51, 44)，以及离中心点最近的两个方格(51, 45)和(52, 44)来预测(见build_targets函数里的解析),
        换句话说这三个方格预测的gt box是同一个，其中心点是(51.7, 44.8)，但tbox保存这三个方格预测的gt box的xy时，保存的是针对这三个方格的偏移量
        分别是：
        (51.7 - 51 = 0.7, 44.8 - 44 = 0.8)
        (51.7 - 51 = 0.7, 44.8 - 45 = -0.2)
        (51.7 - 52 = -0.3, 44.8 - 44 = 0.8)
        """
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            #pi[1,3,80,80,21] pi[...,0] = [1,3,80,80]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                #ps[6,21] [num_targets,] pi[1,3,80,80,21] b[1,6] a[1,6] gj[1,6] gi[1,6]# 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                #这句话是pytorch的高级索引 a[[0,1],[1,2]]表示取出a数组中的a[0,1]和a[1,2]
                #在这里表示在预测的值中取出一个batch中的image_index为b的，使用anchor_index为a的，目标中心点为(gj,gi)的目标
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5#将预测中心的坐标变换到-0.5-1.5之间 [0,1]*2-0.5 = [-0.5,1.5]
                # ps[:, 2:4].sigmoid() * 2) ** 2
                # 的范围是0~4， 再乘以anchors[i]， 表示把预测框的宽和高限制在4倍的anchors内，这是为了解决yolov3和yolov4对预测框宽高无任何约束的问题，这个4和默认的超参数anchor_t是相等的，也是有关联的，调整时建议一起调整。
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                #pxy[6,2] pwh[6,2] pbox[6,4]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #计算预测框6个与gt box iou iou[6]
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  #iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                if self.kpt_label:
                    #Direct kpt prediction
                    # pkpt_x = ps[:, 6::3] * 2. - 0.5
                    # pkpt_y = ps[:, 7::3] * 2. - 0.5
                    # pkpt_score = ps[:, 8::3]
                    pkpt_x = ps[:, 5+self.nc::3] * 2. - 0.5
                    pkpt_y = ps[:, 6+self.nc::3] * 2. - 0.5
                    pkpt_score = ps[:, 7+self.nc::3]

                    #mask
                    kpt_mask = (tkpt[i][:, 0::2] != 0)
                    lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 
                    #l2 distance based loss
                    #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                    #oks based loss
                    d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2
                    s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True)
                    kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                    lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
                # Objectness giou比例gr  detach函数使得iou不可反向传播， clamp将小于0的iou裁剪为0
                # 得到根据iou从小到大排序的image index, anchor index, gridy, gridx, iou。
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5+self.nc], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            #Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkptv *= self.hyp['cls']
        lkpt *= self.hyp['kpt']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkpt + lkptv
        # loss * bs: 整个batch的总损失
        # .detach()  利用损失值进行反向传播 利用梯度信息更新的是损失函数的参数 而对于损失这个值是不需要梯度反向传播的
        return loss * bs, torch.cat((lbox, lobj, lcls, lkpt, lkptv, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        '''
        整体对gt box处理策略：
        1、将gt box复制3份，原因是有三种长宽的anchor， 每种anchor都有gt box与其对应，也就是在筛选之前，一个gt box有三种anchor与其对应。
        2、过滤掉gt box的w和h与anchor的w和h的比值大于设置的超参数anchor_t的gt box。
        3、剩余的gt box，每个gt box使用至少三个方格来预测，一个是gt box中心点所在方格，另两个是中心点离的最近的两个方格

        为compute_loss准备目标，输入目标(image,class,x,y,w,h)
        Args:
            p: p[i]的作用是得到每个feature的shape
            预测框由模型中的三个检测头返回的三个yolo层输出
            tensor格式.list列表存放三个tensor 对应三个yolo层的输出
            eg:[bs,anchor_num,grid_h,grid_w,xywh+class+classesl] -> [4,3,80,80,20][4,3,40,40,20][4,3,20,20,20]
            预测值p是三个yolo层 每个grid（每个grid有三个尺度的预测值）的预测值，后面要进行正样本筛选

            targets:数据增强后的GT框[63.6] [num_target,image_index+class+xywh]xywh为归一化后的框

        Returns:
            tcls：表示这和个target所属的class index
            tbox:xywh 其中xy为这个target对当前的grid左上角的偏移量
            indices：b:该target属于的image index
                    a:该 target 使用的anchor index
                    gj:经过筛选后确定某个target在某个网格中进行预测(计算损失) gj表示这个网格的左上角y坐标
                    gi:表示这个网格的左上角x坐标
            anchor：表示这个target对当前feature map所使用的anchor尺度
        '''
        #num of anchor ,num of target
        na, nt = self.na, targets.shape[0]  # number of anchors, targets[4,16] 16:[image_index+class+xywh+kpt_num(5)*2]
        #存放匹配结果
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        if self.kpt_label:
            # gain = torch.ones(41, device=targets.device)  # normalized to gridspace gain
            #gain是为了将后面targets=[na,nt,16]中的归一化后的xywh映射到相对应的feature mamap
            #gain[1,17]
            gain = torch.ones(self.kpt_num*2 +7, device=targets.device)  # normalized to gridspace gain
        else:
            gain = torch.ones(7, device=targets.device)  #normalized to gridspace gain
        #ai 为anchor索引 四个head 每个head三个anchor 将每个head上anchor编号0 1 2,用view将其按列的形式排列，再用repeat函数复制nt(目标数)列
        """
        [           [
         [0,],        [0,],[0,],[0,],[0,],         
         [1,],   ->   [1,],[1,],[1,],[0,],
         [2,]         [2,][2,],[2,],[0,]
              ]                 ]         
        """
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # target[4,16].repeat(3(na),1,1) ai[3,4,None] -> concat(target[3,4,16] ai[3,4,None])->target[3,4,17]  17:[[image_index+class+xywh+kpt_num(5)*2+anchor_index]
        #对每一个feature map：.repeat(na,1,1)这一步将target复制3(na)份 对应每一个feature map的三个anchor
        #通过将target和前面的ai进行cat拼接操作 就相当于给所有head 都分配了anchor索引0 1 2，后面再过滤
        #至此 target(...,:16) 保存targets信息 target(...,16:)保存对anchor索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #g off用来扩展正样本，因为预测框预测到的target可能不止当前的grid预测到了
        #可能周围的格子也预测到了高质量的样本 将这部分预测信息也加入正样本中
        #
        g = 0.5  # bias 中心偏移 用来衡量target中心点离哪个格子更近
        #以自身 + 周围左上右下四个网格 = 5个网格 用来计算offsets
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        #遍历三个feature 对gt 进行target和anchor的匹配来确定正样本
        for i in range(self.nl): #nl为检测头个数
            #第i个检测头 对应的三个anchor尺寸(相对于feature map) anchors [3,2] [anchor_num,wh]
            anchors = self.anchors[i]
            #
            if self.kpt_label:
                # gain[2:40] = torch.tensor(p[i].shape)[19*[3, 2]]  # xyxy gain
                #gain保存每个输出feature map的宽高 -> gain[2:17] [1,1,80,80,80,80,80,...,80,80,80,80,80,1] imag_index class wywh kpt_num *2 anc_num
                #torch.tensor(shape)=[bs,3,80,80,21] gain 从索引2:16的位置保存feature map 的宽高 选择p.shape中3 和 2位置的值 即 feature map的宽高
                gain[2:self.kpt_num*2 +7 -1] = torch.tensor(p[i].shape)[(2+self.kpt_num)* [3, 2]]  # xyxy gain
            else:
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            #t[3,4,17] * gain[1,17] = [3,4,17] 将target 中的xywh 的归一化尺度放缩到相对当前feature map的坐标尺度
            t = targets * gain
            #开始匹配
            if nt:
                # Matches
                #t = [na,nt,17] t[:,:,4:6]=[na,nt,2]=[3,4,2]
                #achrors[:,none] = [3,1,2] = anchors[:,None,:] 添加一个新轴
                #r = [na,nt,2] = [3,4,2]
                #所有的gt与当前层的三个anchor的宽高比(w/w h/h)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                #筛选条件 GT与anchor比值超过一定阈值 作为负样本
                # j: [3, 4]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                #torch.max(r,1./r)逐元素地取r[3,4,2]和1/r[3,4,2]中的最大值 放于一个张量[3,4,2]中
                #torch.max(r,1./r).max(2) 取上面保存张量[3,4,2]中第三个维度[:,:,2]的最大值及其索引,返回一个元祖(tensor,index)
                #torch.max(r,1./r).max(2)[0] 取元祖中的值的tensor
                #j[3,4]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  #compare
                # v4 v3的筛选方法:wih_iow GT与anchor的wh_iou超过一定阈值就是正样本
                # V4:j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  #V3: iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #布尔掩码
                #t[3,4,17] ->[2,17]
                t = t[j]  # filter

                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                # Offsets
                gxy = t[:, 2:4]  # grid xy  取过滤后的gt box 也就是target中心的坐标xy(相对feature map左上角的坐标)
                gxi = gain[[2, 3]] - gxy  # inverse 将图像左上角为原点的坐标变换为以图像右下角为原点的坐标 得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                #距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j[1,2] k[1,2] 以图像左上角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。
                # j和k的shape都是(1,2),为过滤后的gt box数量，true的位置分别表示靠近方格左边[j]的gt box和靠近方格上方[k]的gt box
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l[2] m[2] 以图像右下角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。
                # l和m的shape都是(2),为过滤后的gt box数量，true的位置分别表示靠近方格右边[l]的gt box和靠近方格下边[m]的gt box
                #l[2] m[2]
                #j和l的值是刚好相反的，k和m的值也是刚好相反的。
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                #j[5,2] torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                # 将j, k, l, m组合成一个tensor，另外还增加了一个全为true的维度。组合之后，j的shape为(5, 2)布尔掩码数组

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # 第一个t是保留所有的gt box，因为上一步里面增加了一个全为true的维度，
                # 第二个t保留了靠近方格左边的gt box，
                # 第三个t保留了靠近方格上方的gt box，
                # 第四个t保留了靠近方格右边的gt box，
                # 第五个t保留了靠近方格下边的gt box，
                # t: [2, 17] -> 复制5份target[5, 2, 17]
                # j: [5, 2]  t: [5, 6, 17][j] -> t[6,17]
                t = t.repeat((5, 1, 1))
                t =t[j]
                # torch.zeros_like(gxy)[None]: [1, 2, 2]   off[:, None]: [5, 1, 2] ->([1,2,2]+[5,1,2] = [5,2,2][j]    => offsets[6,2]
                # j筛选后: [6, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                # offsets的shape为(808, 2), 表示保留下来的808个gtbox的x, y对应的偏移，
                # 第一个t保留所有的gtbox偏移量为[0, 0], 即不做偏移
                # 第二个t保留的靠近方格左边的gtbox，偏移为[0.5, 0]，即向左偏移0.5(后面代码是用gxy - offsets，所以正0.5表示向左偏移)，则偏移到左边方格，表示用左边的方格来预测
                # 第三个t保留的靠近方格上方的gtbox，偏移为[0, 0.5]，即向上偏移0.5，则偏移到上边方格，表示用上边的方格来预测
                # 第四个t保留的靠近方格右边的gtbox，偏移为[-0.5, 0]，即向右偏移0.5，则偏移到右边方格，表示用右边的方格来预测
                # 第五个t保留的靠近方格下边的gtbox，偏移为[0, 0.5]，即向下偏移0.5，则偏移到下边方格，表示用下边的方格来预测
                # 一个gtbox的中心点x坐标要么是靠近方格左边，要么是靠近方格右边，y坐标要么是靠近方格上边，要么是靠近方格下边，
                # 所以一个gtbox在以上五个t里面，会有三个t是true。
                # 也即一个gtbox有三个方格来预测，一个是中心点所在方格，另两个是离的最近的两个方格。而yolov3只使用中心点所在的方格预测，这是与yolov3的区别。
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # t[6,17] [ima_inx+class+xywh+knum*2+anchor_index]
            # b[6,2] c[6,2]
            b, c = t[:, :2].long().T  # image_index, class
            #gxy[6,2] gwh[6,2]
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            #gij [6,2]
            gij = (gxy - offsets).long()#预测真实框的网格所在上下角坐标（有左上右下的网格）
            #gi[1,6] gj[1,6]
            gi, gj = gij.T  # grid xy indices

            # Append 将中心点偏移到相邻最近的方格里，然后向下取整， gij的shape为(6, 2)t[6,17] t[:,-1]= anchor_index
            a = t[:, -1].long()  # anchor indices
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            # clamp_是PyTorch中的一个张量操作，用于限制张量中的值在指定的范围内。
            # 具体来说，clamp_将张量中的每个元素限制在一个指定的范围内，并将超出这个范围的元素设置为边界值。
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # b: image, a:anchor, gj:左上角y坐标 gi:左上角x坐标
            # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box

            if self.kpt_label:
                for kpt in range(self.nkpt):
                    t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6 + 2*kpt: 6+  2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]

                tkpt.append(t[:, 6:-1])
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        #通过上面的样本匹配操作我们会得到4个值.
        #tcls:存储target中的类别, tbox：gt中的box信息, indices: 当前gtbox属于第几张图像, gtbox与anchor的对应关系以及所属的cell坐标.  anchors:anchor信息
        #获得b：image_in; a:anchor_in, gj:gi  Cell的纵坐标与横坐标 tobj是用来后面存储gt中的目标信息,
        return tcls, tbox, tkpt, indices, anch
