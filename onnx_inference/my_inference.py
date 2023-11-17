import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm
from pathlib import Path
import torch
import time
import torchvision
from utils.general import non_max_suppression_export, non_max_suppression
import onnx

parser = argparse.ArgumentParser()
# parser.add_argument("--model-path", type=str, default="yolov5s6_pose_640s.onnx")
parser.add_argument("--model-path", type=str, default="../yolov5s6_pose_640.onnx")
parser.add_argument("--img-path", type=str, default="./sample_ips.txt")
parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
args = parser.parse_args()

# RGB
_CLASS_COLOR_MAP = [
    (0, 0, 255),  # Person (blue).
    (255, 0, 0),  # Bear (red).
    (0, 255, 0),  # Tree (lime).
    (255, 0, 255),  # Bird (fuchsia).
    (0, 255, 255),  # Sky (aqua).
    (255, 255, 0),  # Cat (yellow).
]
# 调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5


def read_img(img_file, img_mean=127.5, img_scale=1 / 127.5):
    # 读取图片，Cv2读取的彩色图片形式为[height,width,3]，其中维度=2上的通道模式通常为BGR,[:,:,::-1]将图片通道转换为RGB
    img = cv2.imread(img_file)[:, :, ::-1]
    # 将图片大小调整为640*640，缩放时采用线性差值INTER_LINEAR来保证图像在大小变换中保持平滑
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    # 对图像进行归一化 将图像中每个像素减去IMG_MEAN 再除以img_scale 通常这步将图像的像素值缩放到一定范围内
    img = (img - img_mean) * img_scale
    # 将图像数据转化为numpy数组，确保数据的数据类型为32位浮点数
    img = np.asarray(img, dtype=np.float32)
    # 在数组的第一个维度上添加一个维度 将图像变成Batch维度中的一个数据
    img = np.expand_dims(img, 0)
    # 将数组维度重新排列，将其变为B C H W的维度
    img = img.transpose(0, 3, 1, 2)
    return img


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    # model_path指向一个onnx格式的权重，onnxruntime.InferenceSession创建了一个ONNX Runtime的推理会话 会话加载指定的ONNX模型，准备推理
    session = onnxruntime.InferenceSession(model_path, None)
    # get_inputs函数获取模型的输入文件名
    input_name = session.get_inputs()[0].name
    output_name = [output.name for output in session.get_outputs()]
    # 将输出存储在output中
    '''
        run(output_names, input_feed, run_options=None)
         作用：计算预测的输出        
         参数:
            output_names – 输出的名称，是一个列表，若为空，则不返回任何特定的张量
            input_feed – 字典 { input_name: input_value }
                ONNX模型通常有一个或者多个输入张量，这些张量在模型推理过程中需要被填充
            run_options – See onnxruntime.RunOptions.
         返回值:
            结果列表, 每个结果要不是一个numpy数组，要不是一个稀疏张量，列表或者字典      
    '''
    # output = session.run([], {input_name: input})
    output = session.run(output_name, {input_name: input})
    return output
def model_inference_image_list(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    # 创建保存图像的路径，如果存在则忽略
    os.makedirs(args.dst_path, exist_ok=True)
    # 打开需要处理的图像文件路径，并转化为list
    img_file_list = list(open(img_path))
    # 返回一个元祖 每个元祖包含一个序列和迭代器的值
    pbar = enumerate(img_file_list)
    max_index = 20
    # 设置进度条
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))

    for img_index, img_file in pbar:
        # 设置进度条
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        # 去除字符串img_file末尾的空格或空字符
        img_file = img_file.rstrip()
        # 输入图像数据
        input = read_img(img_file, mean, scale)
        # 输出模型数据
        output = model_inference(model_path, input)
        # 在dst_path中保存img_file文件 os.path.basename提取路径中的文件名
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        # 后处理文件
        # post_process(img_file, dst_file, output[0], score_threshold=0.3)
        my_process(img_file, dst_file, output, score_threshold=0.3)

def my_process(img_file, dst_file, output_n, score_threshold=0.3):
    #输入为4张量的onnx处理函数
    output = [1,1,1,1]
    for i in range(4):
        output[i] =torch.from_numpy(output_n[i])
    z=[]
    #grid[1,1,1,1] -> {list:3}[tessor[0.],tensor[0.],tensor[0.]]
    grid = [[]]*4
    for i in range(4):
        grid[i] = torch.tensor([torch.zeros(1)])
    na = 3
    no = 57
    stride=torch.tensor([8,16,32,64])
    anchor_grid =[
        [[[[[ 19.,  27.]]],
          [[[ 44.,  40.]]],
          [[[ 38.,  94.]]]]],

        [[[[[ 96.,  68.]]],
          [[[ 86., 152.]]],
          [[[180., 137.]]]]],

        [[[[[140., 301.]]],
          [[[303., 264.]]],
          [[[238., 542.]]]]],

        [[[[[436., 615.]]],
          [[[739., 380.]]],
          [[[925., 792.]]]]]
        ]
    anchor_grid = torch.tensor(anchor_grid)
    for i in range(4):
        x_det = output[i][..., :6]  # x_det -> [1,3,8,8,6]
        x_kpt = output[i][..., 6:]  # x_kpt -> [1,3,8,8,51]
        bs, _, nx, ny, _ = output[i].shape
        if grid[i].shape[2:4] != output[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny).to(output[i].device)
            # grid[i] = _make_grid(nx, ny)
        kpt_grid_x = grid[i][..., 0:1]  # kpt_grid_x:[1,1,8,8,1]
        kpt_grid_y = grid[i][..., 1:2]  # kpt_grid_y:[1,1,8,8,1]
        y = x_det.sigmoid()

        xy = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        wh = (y[..., 2:4] * 2) ** 2 * anchor_grid[i].view(1, na, 1, 1, 2)  # wh
        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * stride[i]  # xy
        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * stride[i]  # xy
        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()
        #y [1,3,80,80,6]->#y[1,3,80,80,57]
        y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)
        # z[[1,19200,57],[1,4800,57],[1,1200,57],[1,300,57]]
        z.append(y.view(bs, -1, no))
    #z[[1,19200,57],[1,4800,57],[1,1200,57],[1,300,57]] -> [1,25500,57]
    z = torch.cat(z, 1)
    z = non_max_suppression(z, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=True, nc=None, nkpt=True)
    post_process(img_file, dst_file, z, score_threshold=0.3)

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Params:
         prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
         conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
         iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
         classes: 是否nms后只保留特定的类别 默认为None
         agnostic: 进行nms是否也去除不同类别之间的框 默认False
         multi_label: 是否是多标签  nc>1  一般是True
         labels:
         max_det: 每张图片的最大目标个数 默认1000
         merge: use merge-NMS 多个bounding box给它们一个权重进行融合  默认False
        nc:类别数
    Returns:
    """
    if nc is None:
        nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
    #xc 用于保存候选预测框 box 置信度大于阈值的位置，值为True 否则为False
    #布尔掩码数组
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    #设置检测框宽高的最大值和最小值
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    #每个图片检测的最多物体
    max_det = 300  # maximum number of detections per image
    #每个图像最多检测物体的个数
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    #nms执行时间的阈值 超过时间就退出
    time_limit = 10.0  # seconds to quit after
    #是否需要冗余的检测detection
    redundant = True  # require redundant detections
    #
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    merge = False  # use merge-NMS

    t = time.time()#记录当前时刻
    #存放最终筛选过后的预测框结果
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    #image index,image inference5
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        #第二层过滤 根据conf_thres滤除目标背景（obj_conf<conf_shres 0.1的目标） x=[26,57]
        #x[25500,57]->[26,57]
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        """
        # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        # Cat apriori labels if autolabelling 自动标注label时调用  一般不用
        # 自动标记在非常高的置信阈值（即 0.90 置信度）下效果最佳,而 mAP 计算依赖于非常低的置信阈值（即 0.001）来正确评估 PR 曲线下的区域。
        # 这个自动标注我觉得应该是一个类似RNN里面的Teacher Forcing的训练机制 就是在训练的时候跟着老师(ground truth)走
        # 但是这样又会造成一个问题: 一直靠老师带的孩子是走不远的 这样的模型因为依赖标签数据,在训练过程中,模型会有较好的效果
        # 但是在测试的时候因为不能得到ground truth的支持, 所以如果目前生成的序列在训练过程中有很大不同, 模型就会变得脆弱。
        # 所以个人认为(个人观点): 应该在下面使用的时候有选择的开启这个trick 比如设置一个概率p随机开启 或者在训练的前n个epoch使用 后面再关闭
        """
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        return output
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def post_process(img_file, dst_file, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    output = output[0].numpy()
    # print(f'output de tyepe is {type(output)},output[0] de type is {type(output)}')
    # yolov5s6_pose_640_ti_lite_54p9_82p2.onnx 输出shape为[2,57] yolov5s6_pose_640.onnx 输出shape为[1.3.80.80.57]
    # [2,57]-> 57 = x_min ,y_min ,x_max ,y_min ,box_conf ,cls_conf 17 *3  三个通道的关键点
    # 将output张量中的值提取出来 box boxconf clsconf kpt
    det_bboxes, det_scores, det_labels, kpts = output[:,:4], output[:,4], output[: ,5], output[:, 6:]
    # 输入提取测试图像
    img = cv2.imread(img_file)
    # To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    height ,width,_ = img.shape
    width = int(width)
    height = int(height)
    path = Path(dst_file)
    suffix = path.suffix

    dst_txt_file = dst_file.replace(suffix, '.txt')

    # 准备写入预测框信息的txt文件

    f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        # len(det_bboxed) = 2
        # 读取 box数据
        det_bbox = det_bboxes[idx]
        # 读取关键点数据
        kpt = kpts[idx]
        for i in range(len(kpt)):
            if i % 3 == 0:
                kpt[i] = (kpt[i] * width) / 640
            if i % 3 == 1:
                kpt[i] = (kpt[i] * height) / 640
        # 如果预测值>0 那么将该预测值的 cls_conf box_conf x_min y_min x_max y_max 写入进txt文件
        if det_scores[idx] > 0:
            f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(det_labels[idx], det_scores[idx],
                                                                               det_bbox[1], det_bbox[0], det_bbox[3],
                                                                               det_bbox[2]))
        # 如果预测置信度值大于置信度阈值
        if det_scores[idx] > score_threshold:
            # 选择指定的颜色
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
            #对图像尺寸不是640*640的图像进行坐标转换
            # start_point = (round(det_bbox[0]*width / 640), round(det_bbox[1]*height / 640))
            # end_point = (round(det_bbox[2] * width / 640), round(det_bbox[3] * height / 640))
            start_point = (round(det_bbox[0]*width / 640), round(det_bbox[1]*height / 640))
            end_point = (round(det_bbox[2] * width / 640), round(det_bbox[3] * height / 640))
            img = cv2.rectangle(img, start_point, end_point,color_map[::-1], 2)
            cv2.putText(img, "id:{}".format(int(det_labels[idx])), (int(det_bbox[0]*width / 640 + 5), int(det_bbox[1]*width / 640) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0]*width / 640 + 5), int(det_bbox[1]*width / 640) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            '''
            作用：
                在指定的图片上画框
                 cv2.rectangle(image, start_point, end_point, color, thickness)
                参数：
                    image：指定画框的图片
                    start_point:框的开始点，一般用一个元祖(x,y)表示
                    end_point:框的结束点，一般用一个元祖(x,y)表示
                    color：框的颜色 用BGR表示
                    thickness：用多少像素(px)画框，如果=-1，用指定的颜色填满画的框
            '''
                #在图像上画框,cv2.rectangle根据预测框的左上和右下角坐标进行画框，color_map[::-1]表示颜色用GBR表示画出，2表示检测框的宽度为2像素
            # img = cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
            #                     color_map[::-1], 2)
            # 在图像上写字，要写的字 写的字体的左上角坐标 字体 字体大小 颜色 宽度
            # cv2.putText(img, "id:{}".format(int(det_labels[idx])), (int(det_bbox[0] + 5), int(det_bbox[1]) + 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            # cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0] + 5), int(det_bbox[1]) + 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            # 在图像上画关键点
            plot_skeleton_kpts(img, kpt)
    # 将画好框的img写入指定文件
    cv2.imwrite(dst_file, img)
    f.close()


def plot_skeleton_kpts(im, kpts, steps=3):
    # 每个通道层的关键点数 = 总的kpts /层数 -> num_kpts = 51 /3
    num_kpts = len(kpts) // steps
    # plot keypoints
    # 一共画17个点 kid = 0-16
    for kid in range(num_kpts):
        # 取rgb三通道颜色值
        r, g, b = pose_kpt_color[kid]
        # 取关键点的坐标 x y
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        # 取关键点的置信度 conf
        conf = kpts[steps * kid + 2]
        # 如果关键点置信度大于0.5 则将关键点画在图像上
        if conf > 0.5:  # Confidence of a keypoint has to be greater than 0.5
            # 根据中心点在图像上画圆
            # 需要画的图像/圆点坐标/半径/圆圈的颜色/圆圈宽度 -1表示用指定颜色填满圆圈
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    # plot skeleton
    # 画出关键点之间的骨架
    for sk_id, sk in enumerate(skeleton):
        '''
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        radius = 5
    '''
        # 指定骨架连线的颜色
        r, g, b = pose_limb_color[sk_id]
        # -1是为了适配元祖的下标 取出两个关键点之间的坐标
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:  # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

def main():
    model_inference_image_list(model_path=args.model_path, img_path=args.img_path,
                               mean=0.0, scale=0.00392156862745098,
                               dst_path=args.dst_path)


if __name__ == "__main__":
    main()
